"""
Run HybridECDet TensorRT engine on image / video / webcam.

Install:
    pip install tensorrt tensorrt-cu12 cuda-python --extra-index-url https://pypi.nvidia.com

Usage:
    python src/infer_tensorrt.py --engine model_drone_fp16.trt --image  path/to/image.jpg
    python src/infer_tensorrt.py --engine model_drone_fp16.trt --video  path/to/video.mp4 --output result.mp4
    python src/infer_tensorrt.py --engine model_drone_fp16.trt --webcam 0
"""
import argparse
import os
import time

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

# Reuse preprocessing / post-processing / visualization from infer_onnx.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from infer_onnx import (
    preprocess, postprocess,
    draw_detections, draw_hud, draw_heatmap,
    VISDRONE_CLASSES, _open_writer,
)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ── TRT session ────────────────────────────────────────────────────────────────

class TRTSession:
    """Loads a .trt engine and runs inference with cuda-python buffers."""

    def __init__(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # build name → (host_buf, device_buf, shape, dtype) maps
        self.inputs  = {}
        self.outputs = {}
        self._bindings = []

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

            h_buf = np.empty(shape, dtype=dtype)
            err, d_buf = cudart.cudaMalloc(nbytes)
            assert err == cudart.cudaError_t.cudaSuccess, f"cudaMalloc failed: {err}"

            self.context.set_tensor_address(name, d_buf)
            entry = (h_buf, d_buf, shape, dtype)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name]  = entry
            else:
                self.outputs[name] = entry

        err, self.stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

        # print I/O summary
        for name, (h, d, shape, dtype) in self.inputs.items():
            print(f"[trt]    input  '{name}'  {shape}  {dtype}")
        for name, (h, d, shape, dtype) in self.outputs.items():
            print(f"[trt]    output '{name}'  {shape}  {dtype}")

    def run(self, **feed: np.ndarray) -> dict[str, np.ndarray]:
        """feed: {input_name: numpy_array}  →  {output_name: numpy_array}"""
        # copy inputs H→D
        for name, arr in feed.items():
            h_buf, d_buf, shape, dtype = self.inputs[name]
            arr = np.ascontiguousarray(arr.astype(dtype))
            cudart.cudaMemcpyAsync(
                d_buf, arr.ctypes.data, arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
            )

        self.context.execute_async_v3(self.stream)

        # copy outputs D→H
        results = {}
        for name, (h_buf, d_buf, shape, dtype) in self.outputs.items():
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data, d_buf, h_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream
            )
            results[name] = h_buf

        cudart.cudaStreamSynchronize(self.stream)
        return results

    def __del__(self):
        for _, (_, d_buf, _, _) in {**self.inputs, **self.outputs}.items():
            cudart.cudaFree(d_buf)
        cudart.cudaStreamDestroy(self.stream)


# ── Inference helper ───────────────────────────────────────────────────────────

def infer_trt(sess: TRTSession, img_bgr: np.ndarray,
              input_h: int, input_w: int,
              conf_thr: float, iou_thr: float):
    tensor, ratio, dw, dh = preprocess(img_bgr, input_h, input_w)
    results = sess.run(image=tensor)

    # output names match export_onnx.py: 'boxes', 'scores', 'hm'
    boxes_out  = results['boxes']    # (1, K, 4)
    scores_out = results['scores']   # (1, K, C)
    hm_out     = results['hm']       # (1, C, H/4, W/4)

    dets = postprocess(
        boxes_out[0], scores_out[0],
        input_w, input_h, ratio, dw, dh,
        conf_thr, iou_thr,
    )
    return dets, hm_out[0], ratio, dw, dh


# ── Mode runners ───────────────────────────────────────────────────────────────

def run_image(args, sess: TRTSession):
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    print(f"[image]  {args.image}  {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    dets, hm_np, *_ = infer_trt(sess, img_bgr, args.input_h, args.input_w,
                                  args.conf, args.iou)
    print(f"[detect] {len(dets)} objects")
    for d in dets:
        name = VISDRONE_CLASSES.get(d['cls'], f"cls{d['cls']}")
        x1, y1, x2, y2 = [int(v) for v in d['box']]
        print(f"         {name:<18} score={d['score']:.3f}  [{x1},{y1},{x2},{y2}]")

    vis = draw_detections(img_bgr, dets)
    draw_hud(vis, len(dets))
    cv2.imwrite(args.output, vis)
    print(f"[saved]  {args.output}")

    if not args.no_heatmap:
        cv2.imwrite(args.output_hm, draw_heatmap(img_bgr, hm_np))
        print(f"[saved]  {args.output_hm}")


def run_video(args, sess: TRTSession):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.video}")

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video]  {src_w}x{src_h}  {src_fps:.1f} fps  ~{n_frames} frames")

    out_path = args.output if args.output != 'result.jpg' else 'result.mp4'
    writer, out_path = _open_writer(out_path, src_fps, src_w, src_h)

    frame_id   = 0
    t_total    = 0.0
    fps_smooth = src_fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            dets, hm_np, *_ = infer_trt(sess, frame, args.input_h, args.input_w,
                                          args.conf, args.iou)
            elapsed    = time.perf_counter() - t0
            t_total   += elapsed
            fps_smooth = 0.9 * fps_smooth + 0.1 / (elapsed + 1e-6)

            vis = draw_detections(frame, dets)
            draw_hud(vis, len(dets), fps=fps_smooth, frame_id=frame_id)
            writer.write(vis)

            frame_id += 1
            if frame_id % 50 == 0 or frame_id == 1:
                pct = f"{100*frame_id/n_frames:.0f}%" if n_frames > 0 else "?"
                print(f"[video]  frame {frame_id}/{n_frames} ({pct})  "
                      f"fps={fps_smooth:.1f}  dets={len(dets)}")
    finally:
        cap.release()
        writer.release()

    print(f"[done]   {frame_id} frames  avg={frame_id/t_total:.1f} fps  → {out_path}")


def run_webcam(args, sess: TRTSession):
    cap = cv2.VideoCapture(int(args.webcam))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam id={args.webcam}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[webcam] camera={args.webcam}  {src_w}x{src_h}  press q/ESC to quit, s to snapshot")

    writer    = None
    save_path = None
    if args.save_webcam:
        save_path        = args.output if args.output != 'result.jpg' else 'webcam_out.mp4'
        cam_fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer, save_path = _open_writer(save_path, cam_fps, src_w, src_h)

    frame_id   = 0
    fps_smooth = 25.0
    snap_idx   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            dets, *_ = infer_trt(sess, frame, args.input_h, args.input_w,
                                  args.conf, args.iou)
            elapsed    = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 / (elapsed + 1e-6)

            vis = draw_detections(frame, dets)
            draw_hud(vis, len(dets), fps=fps_smooth, frame_id=frame_id)

            if writer is not None:
                writer.write(vis)

            cv2.imshow("HybridECDet TRT — webcam  (q/ESC quit  s snapshot)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                snap = f"snapshot_{snap_idx:04d}.jpg"
                cv2.imwrite(snap, vis)
                print(f"[snap]   {snap}")
                snap_idx += 1

            frame_id += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"[done]   {frame_id} frames  fps~{fps_smooth:.1f}")
    if save_path:
        print(f"[saved]  {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='HybridECDet TensorRT inference')
    p.add_argument('--engine',     required=True,  help='Path to .trt engine file')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',  help='Input image (image mode)')
    src.add_argument('--video',  help='Input video (video mode)')
    src.add_argument('--webcam', metavar='CAM_ID', help='Webcam device id, e.g. 0')

    p.add_argument('--output',      default='result.jpg')
    p.add_argument('--output_hm',   default='heatmap.jpg')
    p.add_argument('--input_w',     type=int,   default=832)
    p.add_argument('--input_h',     type=int,   default=512)
    p.add_argument('--conf',        type=float, default=0.3)
    p.add_argument('--iou',         type=float, default=0.45)
    p.add_argument('--no_heatmap',  action='store_true')
    p.add_argument('--save_webcam', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[trt]    version={trt.__version__}  engine={args.engine}")
    sess = TRTSession(args.engine)

    if args.image:
        run_image(args, sess)
    elif args.video:
        run_video(args, sess)
    else:
        run_webcam(args, sess)


if __name__ == '__main__':
    main()
