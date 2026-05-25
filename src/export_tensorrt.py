"""
Convert ONNX → TensorRT engine (pip-installed tensorrt).

Install:
    pip install tensorrt pycuda --extra-index-url https://pypi.nvidia.com
    # pycuda also needs: pip install pycuda

Usage:
    # FP32
    python src/export_tensorrt.py --onnx model_drone.onnx --output model_drone_fp32.trt

    # FP16  (recommended — 2x faster, minimal accuracy drop)
    python src/export_tensorrt.py --onnx model_drone.onnx --output model_drone_fp16.trt --fp16

    # INT8  (fastest — requires calibration images)
    python src/export_tensorrt.py --onnx model_drone.onnx --output model_drone_int8.trt --int8 \
        --calib_dir path/to/calib_images/
"""
import argparse
import os
import glob
import sys
import time
import threading

# ── GPU selection must happen before pycuda/tensorrt are imported ──────────────
# Parse only --gpu_id early so CUDA_VISIBLE_DEVICES is set in time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu_id', type=int, default=0)
_gpu_id = _pre.parse_known_args()[0].gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)

import numpy as np
import pycuda.autoinit  # noqa: F401 — must come before tensorrt, respects CUDA_VISIBLE_DEVICES
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER       = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER_QUIET = trt.Logger(trt.Logger.INTERNAL_ERROR)  # hides scale/tactic warnings during build


# ── Progress helpers ───────────────────────────────────────────────────────────

class _Spinner:
    """Shows a spinning indicator + elapsed time while a blocking call runs."""
    def __init__(self, msg: str):
        self._msg   = msg
        self._stop  = threading.Event()
        self._t0    = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            m, s    = divmod(int(elapsed), 60)
            sys.stdout.write(f"\r{frames[i % len(frames)]}  {self._msg}  [{m:02d}:{s:02d}]  ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)

    def __enter__(self):
        self._t0 = time.time()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        elapsed = time.time() - self._t0
        m, s    = divmod(int(elapsed), 60)
        sys.stdout.write(f"\r✓  {self._msg}  [{m:02d}:{s:02d}]\n")
        sys.stdout.flush()


# ── INT8 Calibrator ────────────────────────────────────────────────────────────

class ImageCalibrator(trt.IInt8MinMaxCalibrator):
    """
    Feeds real images to TensorRT for INT8 calibration.
    Provide 100–500 representative images from your dataset.
    """
    def __init__(self, image_dir: str, input_h: int, input_w: int,
                 batch_size: int = 4, cache_file: str = 'calib_cache.bin',
                 max_images: int = 0):
        super().__init__()
        self.input_h    = input_h
        self.input_w    = input_w
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.index      = 0
        self._cuda      = cuda

        import cv2
        self._cv2 = cv2

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.images = []
        for ext in exts:
            self.images += glob.glob(os.path.join(image_dir, '**', ext), recursive=True)
        if not self.images:
            raise FileNotFoundError(f"No images found in {image_dir} (searched recursively)")
        self.images.sort()

        if max_images > 0 and len(self.images) > max_images:
            step = len(self.images) // max_images   # evenly spaced, covers full dataset
            self.images = self.images[::step][:max_images]
            print(f"[calib]  limited to {len(self.images)}/{len(self.images)*step} images  batch={batch_size}")
        else:
            print(f"[calib]  {len(self.images)} images  batch={batch_size}")

        nbytes = batch_size * 3 * input_h * input_w * 4   # float32
        self.device_buf = cuda.mem_alloc(nbytes)
        self._t_start   = time.time()

    def _preprocess(self, path: str) -> np.ndarray:
        MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
        STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
        img  = self._cv2.imread(path)
        if img is None:
            return np.zeros((3, self.input_h, self.input_w), np.float32)
        img  = self._cv2.resize(img, (self.input_w, self.input_h))
        img  = img[:, :, ::-1].astype(np.float32) / 255.0
        return ((img.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.images):
            return None
        batch = []
        for i in range(self.batch_size):
            idx = (self.index + i) % len(self.images)
            batch.append(self._preprocess(self.images[idx]))
        self.index += self.batch_size

        arr = np.ascontiguousarray(np.stack(batch))
        self._cuda.memcpy_htod(self.device_buf, arr)

        total    = len(self.images)
        pct      = min(self.index, total) * 100 // total
        filled   = pct // 5
        bar      = '█' * filled + '░' * (20 - filled)
        elapsed  = time.time() - self._t_start
        eta      = (elapsed / self.index * (total - self.index)) if self.index < total else 0
        sys.stdout.write(
            f"\r[calib]  |{bar}| {pct:3d}%  "
            f"{min(self.index, total)}/{total} imgs  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
        )
        sys.stdout.flush()
        return [self.device_buf]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[calib]  loading cache {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        elapsed = time.time() - self._t_start
        print(f"\n[calib]  done in {elapsed:.0f}s  cache saved → {self.cache_file}")


# ── Build engine ───────────────────────────────────────────────────────────────

def build_engine(onnx_path: str, fp16: bool, int8: bool,
                 workspace_gb: int, calib_dir: str,
                 input_h: int, input_w: int,
                 calib_limit: int = 0,
                 verbose: bool = False) -> bytes:

    logger  = TRT_LOGGER if verbose else TRT_LOGGER_QUIET
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()

    # ── Parse ONNX ──────────────────────────────────────────────────────────
    print(f"[parse]  {onnx_path}")
    with open(onnx_path, 'rb') as f:
        ok = parser.parse(f.read())
    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    # ── Print network I/O ───────────────────────────────────────────────────
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"[input]  {t.name}  {tuple(t.shape)}  {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"[output] {t.name}  {tuple(t.shape)}  {t.dtype}")

    # ── Builder config ──────────────────────────────────────────────────────
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
    )

    # FP16 is always enabled when INT8 is requested so that layers which
    # cannot be quantized (LayerNorm, TopK, cast ops …) fall back to FP16
    # instead of FP32, which eliminates the "Missing scale and zero-point"
    # warnings and keeps those layers fast.
    if fp16 or int8:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[build]  FP16 enabled")
        else:
            print("[build]  WARNING: GPU does not support fast FP16")

    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = ImageCalibrator(
                calib_dir, input_h, input_w, batch_size=4,
                max_images=calib_limit,
            )
            print("[build]  INT8 enabled  (unsupported layers → FP16 fallback)")
        else:
            print("[build]  WARNING: GPU does not support INT8, using FP16 only")

    # ── Build ───────────────────────────────────────────────────────────────
    with _Spinner("TensorRT optimizing engine (may take several minutes)"):
        serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed — check TRT_LOGGER output above")
    print(f"[build]  engine size={serialized.nbytes/1e6:.1f} MB")
    return serialized


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='ONNX → TensorRT conversion')
    p.add_argument('--onnx',      required=True,          help='Input .onnx file')
    p.add_argument('--output',    default='model.trt',    help='Output .trt engine file')
    p.add_argument('--fp16',      action='store_true',    help='Enable FP16 precision')
    p.add_argument('--int8',      action='store_true',    help='Enable INT8 precision')
    p.add_argument('--calib_dir',   default='calib_images', help='Images dir for INT8 calibration')
    p.add_argument('--calib_limit', type=int, default=500,
                   help='Max images for INT8 calibration (0 = use all, default 500)')
    p.add_argument('--workspace', type=int, default=4,   help='GPU workspace in GB (default 4)')
    p.add_argument('--input_w',   type=int, default=832)
    p.add_argument('--input_h',   type=int, default=512)
    p.add_argument('--gpu_id',    type=int, default=0,  help='GPU device id (default 0)')
    p.add_argument('--verbose',   action='store_true',
                   help='Show all TRT warnings (default: hide scale/zero-point warnings)')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX not found: {args.onnx}")
    if args.int8 and not os.path.isdir(args.calib_dir):
        raise FileNotFoundError(f"calib_dir not found: {args.calib_dir}")

    # ── Show GPU info ───────────────────────────────────────────────────────
    device = cuda.Device(0)   # always 0 after CUDA_VISIBLE_DEVICES remapping
    print(f"[gpu]    id={args.gpu_id}  {device.name()}  "
          f"VRAM={device.total_memory()//1024**2} MB")

    if not args.verbose:
        print("[build]  TRT warnings suppressed — add --verbose to see them")

    serialized = build_engine(
        onnx_path   = args.onnx,
        fp16        = args.fp16,
        int8        = args.int8,
        workspace_gb= args.workspace,
        calib_dir   = args.calib_dir,
        input_h     = args.input_h,
        input_w     = args.input_w,
        calib_limit = args.calib_limit,
        verbose     = args.verbose,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(memoryview(serialized))
    print(f"[saved]  {args.output}")


if __name__ == '__main__':
    main()
