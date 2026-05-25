"""
TensorRT inference on a drone video.

Usage:
    python src/infer_trt_video.py \
        --engine model_drone_fp16.trt \
        --video  test_video/drone.mp4 \
        --output test_video/result.mp4 \
        --conf   0.3
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── GPU selection before pycuda/tensorrt import ────────────────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu_id', type=int, default=0)
os.environ['CUDA_VISIBLE_DEVICES'] = str(_pre.parse_known_args()[0].gpu_id)

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# ── VisDrone classes ───────────────────────────────────────────────────────────
CLASSES = {
    0: 'pedestrian', 1: 'people',    2: 'bicycle', 3: 'car',
    4: 'van',        5: 'truck',     6: 'tricycle', 7: 'awning-tricycle',
    8: 'bus',        9: 'motor',
}
PALETTE = [
    (255,128,0),(255,200,0),(50,220,50),(0,160,255),(180,0,220),
    (220,40,40),(255,60,180),(0,200,200),(255,255,0),(100,100,255),
]

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ── TRT engine ─────────────────────────────────────────────────────────────────

class TRTEngine:
    def __init__(self, engine_path: str):
        with open(engine_path, 'rb') as f:
            self.engine  = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.inputs, self.outputs = {}, {}

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            h_buf = np.empty(shape, dtype=dtype)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self.context.set_tensor_address(name, d_buf)
            store = (h_buf, d_buf)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name]  = store
            else:
                self.outputs[name] = store

    def infer(self, image: np.ndarray) -> tuple:
        """image: (1,3,H,W) float32  →  (boxes, scores, hm)"""
        h_in, d_in = next(iter(self.inputs.values()))
        cuda.memcpy_htod_async(d_in, np.ascontiguousarray(image), self.stream)
        self.context.execute_async_v3(self.stream.handle)

        results = {}
        for name, (h_buf, d_buf) in self.outputs.items():
            cuda.memcpy_dtoh_async(h_buf, d_buf, self.stream)
            results[name] = h_buf
        self.stream.synchronize()

        return results['boxes'][0], results['scores'][0], results['hm'][0]


# ── Pre / post-processing ──────────────────────────────────────────────────────

def preprocess(img: np.ndarray, h: int, w: int):
    ih, iw = img.shape[:2]
    ratio  = min(h / ih, w / iw)
    new_w, new_h = round(iw * ratio), round(ih * ratio)
    dw, dh = (w - new_w) * 0.5, (h - new_h) * 0.5
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.full((h, w, 3), 127.5, np.float32)
    t, l    = round(dh - 0.1), round(dw - 0.1)
    canvas[t:t+new_h, l:l+new_w] = resized
    rgb     = canvas[:, :, ::-1] / 255.0
    tensor  = ((rgb - MEAN) / STD).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return tensor, ratio, dw, dh


def postprocess(boxes, scores, input_w, input_h, ratio, dw, dh, conf, iou):
    max_s = scores.max(axis=1)
    cls   = scores.argmax(axis=1)
    mask  = max_s >= conf
    if not mask.any():
        return []

    b, s, c = boxes[mask], max_s[mask], cls[mask]
    cx = b[:,0]*input_w;  cy = b[:,1]*input_h
    bw = b[:,2]*input_w;  bh = b[:,3]*input_h
    x1 = (cx - bw/2 - dw) / ratio;  y1 = (cy - bh/2 - dh) / ratio
    x2 = (cx + bw/2 - dw) / ratio;  y2 = (cy + bh/2 - dh) / ratio
    xyxy = np.stack([x1,y1,x2,y2], 1)

    # NMS
    areas = (x2-x1).clip(0) * (y2-y1).clip(0)
    order = s.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]; keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (ix2-ix1).clip(0)*(iy2-iy1).clip(0)
        order = order[1:][inter/(areas[i]+areas[order[1:]]-inter+1e-6) <= iou]

    return [{'box': xyxy[i].tolist(), 'score': float(s[i]), 'cls': int(c[i])}
            for i in keep]


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw(frame, dets, fps, fid):
    for d in dets:
        x1,y1,x2,y2 = [int(v) for v in d['box']]
        color = PALETTE[d['cls'] % len(PALETTE)]
        label = f"{CLASSES.get(d['cls'], d['cls'])} {d['score']:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, max(y1-th-6,0)), (x1+tw+6, max(y1,th+6)), color, -1)
        cv2.putText(frame, label, (x1+3, max(y1-3, th+3)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    hud = f"Frame {fid}  |  {fps:.1f} FPS  |  {len(dets)} objs"
    cv2.rectangle(frame, (0,0), (len(hud)*9+10, 28), (20,20,20), -1)
    cv2.putText(frame, hud, (6,20), cv2.FONT_HERSHEY_DUPLEX,
                0.55, (255,255,255), 1, cv2.LINE_AA)
    return frame


def open_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    for codec, ext in [('avc1','.mp4'),('mp4v','.mp4'),('XVID','.avi')]:
        p = path if path.endswith(('.mp4','.avi')) else path + ext
        w_ = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if w_.isOpened():
            print(f"[writer] codec={codec}  → {p}")
            return w_, p
        w_.release()
    raise RuntimeError("No working video codec found")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--engine',   required=True)
    p.add_argument('--video',    required=True)
    p.add_argument('--output',   default='result.mp4')
    p.add_argument('--input_w',  type=int,   default=832)
    p.add_argument('--input_h',  type=int,   default=512)
    p.add_argument('--conf',     type=float, default=0.3)
    p.add_argument('--iou',      type=float, default=0.45)
    p.add_argument('--gpu_id',   type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[engine] loading {args.engine} ...")
    engine = TRTEngine(args.engine)
    print(f"[engine] ready  gpu={args.gpu_id}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.video}")

    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video]  {args.video}  {src_w}x{src_h}  {fps_src:.0f}fps  {n_total} frames")

    writer, out_path = open_writer(args.output, fps_src, src_w, src_h)

    fps_smooth = fps_src
    fid        = 0
    t_total    = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            tensor, ratio, dw, dh = preprocess(frame, args.input_h, args.input_w)
            boxes, scores, hm     = engine.infer(tensor)
            dets = postprocess(boxes, scores, args.input_w, args.input_h,
                               ratio, dw, dh, args.conf, args.iou)

            elapsed    = time.perf_counter() - t0
            t_total   += elapsed
            fps_smooth = 0.9 * fps_smooth + 0.1 / (elapsed + 1e-6)

            draw(frame, dets, fps_smooth, fid)
            writer.write(frame)
            fid += 1

            if fid % 30 == 0 or fid == 1:
                pct = f"{100*fid//n_total}%" if n_total else "?"
                print(f"[run]    {fid}/{n_total} ({pct})  "
                      f"fps={fps_smooth:.1f}  dets={len(dets)}")
    finally:
        cap.release()
        writer.release()

    print(f"[done]   {fid} frames  avg={fid/t_total:.1f} fps")
    print(f"[saved]  {out_path}")


if __name__ == '__main__':
    main()
