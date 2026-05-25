"""
Run HybridECDet ONNX model on a single image, video, or webcam.

Image mode:
    python src/infer_onnx.py \
        --model   model.onnx \
        --image   path/to/image.jpg \
        --conf    0.3 --iou 0.45

Video mode:
    python src/infer_onnx.py \
        --model   model.onnx \
        --video   path/to/video.mp4 \
        --output  result.mp4 \
        --conf    0.3 --iou 0.45

Webcam mode:
    python src/infer_onnx.py \
        --model   model.onnx \
        --webcam  0 \
        --conf    0.3 --iou 0.45
    Press 'q' or ESC to quit. Press 's' to save a snapshot.
    Add --save_webcam to record the whole session to webcam_out.mp4.

Common options:
    --input_w 832 --input_h 512   (must match export resolution)
    --gpu                          use CUDAExecutionProvider
    --no_heatmap                   skip heatmap output (image mode only)
"""
import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# ── VisDrone class names ───────────────────────────────────────────────────────
VISDRONE_CLASSES = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor',
}

_PALETTE = [
    (255, 128,   0),
    (255, 200,   0),
    ( 50, 220,  50),
    (  0, 160, 255),
    (180,   0, 220),
    (220,  40,  40),
    (255,  60, 180),
    (  0, 200, 200),
    (255, 255,   0),
    (100, 100, 255),
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, height: int, width: int, color=(127.5, 127.5, 127.5)):
    h0, w0 = img.shape[:2]
    ratio  = min(height / h0, width / w0)
    new_w  = round(w0 * ratio)
    new_h  = round(h0 * ratio)
    dw     = (width  - new_w) * 0.5
    dh     = (height - new_h) * 0.5
    top,  bottom = round(dh - 0.1), round(dh + 0.1)
    left, right  = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


def preprocess(img_bgr: np.ndarray, input_h: int, input_w: int):
    img_lb, ratio, dw, dh = letterbox(img_bgr, input_h, input_w)
    img_rgb  = img_lb[:, :, ::-1].astype(np.float32) / 255.0
    img_norm = (img_rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor   = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return tensor, ratio, dw, dh


# ── Post-processing ────────────────────────────────────────────────────────────

def cxcywh_to_xyxy(boxes: np.ndarray, input_w: int, input_h: int,
                   ratio: float, dw: float, dh: float) -> np.ndarray:
    cx = boxes[:, 0] * input_w
    cy = boxes[:, 1] * input_h
    w  = boxes[:, 2] * input_w
    h  = boxes[:, 3] * input_h
    x1 = (cx - w / 2 - dw) / ratio
    y1 = (cy - h / 2 - dh) / ratio
    x2 = (cx + w / 2 - dw) / ratio
    y2 = (cy + h / 2 - dh) / ratio
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]
        keep.append(i)
        ix1   = np.maximum(x1[i], x1[order[1:]])
        iy1   = np.maximum(y1[i], y1[order[1:]])
        ix2   = np.minimum(x2[i], x2[order[1:]])
        iy2   = np.minimum(y2[i], y2[order[1:]])
        inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def postprocess(boxes_norm: np.ndarray, scores_sigmoid: np.ndarray,
                input_w: int, input_h: int,
                ratio: float, dw: float, dh: float,
                conf_thr: float, iou_thr: float) -> list:
    max_scores = scores_sigmoid.max(axis=1)
    cls_ids    = scores_sigmoid.argmax(axis=1)
    mask = max_scores >= conf_thr
    if not mask.any():
        return []
    boxes_filt  = boxes_norm[mask]
    scores_filt = max_scores[mask]
    cls_filt    = cls_ids[mask]
    xyxy = cxcywh_to_xyxy(boxes_filt, input_w, input_h, ratio, dw, dh)
    keep = nms(xyxy, scores_filt, iou_thr)
    return [{'box': xyxy[i].tolist(), 'score': float(scores_filt[i]),
             'cls': int(cls_filt[i])} for i in keep]


def infer(sess, in_name: str, img_bgr: np.ndarray,
          input_h: int, input_w: int, conf_thr: float, iou_thr: float):
    """Run one frame end-to-end. Returns (detections, hm_np, ratio, dw, dh)."""
    tensor, ratio, dw, dh = preprocess(img_bgr, input_h, input_w)
    boxes_out, scores_out, hm_out = sess.run(None, {in_name: tensor})
    dets = postprocess(boxes_out[0], scores_out[0],
                       input_w, input_h, ratio, dw, dh,
                       conf_thr, iou_thr)
    return dets, hm_out[0], ratio, dw, dh


# ── Visualization ──────────────────────────────────────────────────────────────

def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    vis = img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        cls   = det['cls']
        score = det['score']
        color = _PALETTE[cls % len(_PALETTE)]
        label = f"{VISDRONE_CLASSES.get(cls, f'cls{cls}')}  {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        lx1, ly1 = x1, max(y1 - th - 8, 0)
        lx2, ly2 = x1 + tw + 8, max(y1, th + 8)
        overlay = vis.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
        cv2.putText(vis, label, (lx1 + 4, ly2 - 4),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_hud(img: np.ndarray, n_dets: int,
             fps: float | None = None, frame_id: int | None = None) -> np.ndarray:
    """Top-left HUD: detection count + optional FPS / frame number."""
    lines = [f"Detections: {n_dets}"]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    if frame_id is not None:
        lines.append(f"Frame: {frame_id}")

    font   = cv2.FONT_HERSHEY_DUPLEX
    scale  = 0.55
    thick  = 1
    pad    = 6
    line_h = 20
    panel_w = 160
    panel_h = pad + line_h * len(lines) + pad

    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    for i, txt in enumerate(lines):
        y = 8 + pad + line_h * (i + 1) - 4
        cv2.putText(img, txt, (14, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def draw_heatmap(img: np.ndarray, hm: np.ndarray) -> np.ndarray:
    hm_max   = hm.max(axis=0)
    hm_u8    = (hm_max * 255).clip(0, 255).astype(np.uint8)
    hm_big   = cv2.resize(hm_u8, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_big, cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(img, 0.5, hm_color, 0.5, 0)
    cv2.putText(overlay, "Stage-1 Heatmap", (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


# ── Mode runners ───────────────────────────────────────────────────────────────

def run_image(args, sess, in_name: str):
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    print(f"[image]  {args.image}  {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    dets, hm_np, ratio, dw, dh = infer(sess, in_name, img_bgr,
                                        args.input_h, args.input_w,
                                        args.conf, args.iou)
    print(f"[detect] {len(dets)} objects")
    for d in dets:
        cls_name = VISDRONE_CLASSES.get(d['cls'], f"cls{d['cls']}")
        x1, y1, x2, y2 = [int(v) for v in d['box']]
        print(f"         {cls_name:<18} score={d['score']:.3f}  [{x1},{y1},{x2},{y2}]")

    vis = draw_detections(img_bgr, dets)
    draw_hud(vis, len(dets))
    cv2.imwrite(args.output, vis)
    print(f"[saved]  {args.output}")

    if not args.no_heatmap:
        hm_vis = draw_heatmap(img_bgr, hm_np)
        cv2.imwrite(args.output_hm, hm_vis)
        print(f"[saved]  {args.output_hm}")


def _open_writer(out_path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    """Try codecs in order until one opens successfully."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # avc1 = H.264 (macOS native), mp4v = MPEG-4, XVID = fallback AVI
    candidates = [
        (out_path,                       'avc1'),
        (out_path,                       'mp4v'),
        (out_path.replace('.mp4', '.avi'), 'XVID'),
    ]
    for path, codec in candidates:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            print(f"[writer] codec={codec}  → {path}")
            return writer, path
        writer.release()
    raise RuntimeError(
        "No working video codec found. Install opencv-python with ffmpeg support, "
        "or try: pip install opencv-python-headless"
    )


def run_video(args, sess, in_name: str):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video]  {args.video}  {src_w}x{src_h}  {src_fps:.1f} fps  ~{n_frames} frames")

    out_path   = args.output if args.output != 'result.jpg' else 'result.mp4'
    writer, out_path = _open_writer(out_path, src_fps, src_w, src_h)

    frame_id  = 0
    t_total   = 0.0
    fps_smooth = src_fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0   = time.perf_counter()
            dets, hm_np, *_ = infer(sess, in_name, frame,
                                    args.input_h, args.input_w,
                                    args.conf, args.iou)
            t1   = time.perf_counter()

            elapsed   = t1 - t0
            t_total  += elapsed
            # exponential moving average for stable FPS display
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / (elapsed + 1e-6))

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

    avg_fps = frame_id / t_total if t_total > 0 else 0
    print(f"[done]   {frame_id} frames  avg={avg_fps:.1f} fps  saved → {out_path}")


def run_webcam(args, sess, in_name: str):
    cam_id = int(args.webcam)
    cap    = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam id={cam_id}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[webcam] camera={cam_id}  {src_w}x{src_h}  press q/ESC to quit, s to snapshot")

    writer      = None
    save_path   = None
    if args.save_webcam:
        save_path  = args.output if args.output != 'result.jpg' else 'webcam_out.mp4'
        cam_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer, save_path = _open_writer(save_path, cam_fps, src_w, src_h)
        print(f"[webcam] recording → {save_path}")

    frame_id   = 0
    fps_smooth = 25.0
    snap_idx   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[webcam] no frame received — camera disconnected?")
                break

            t0 = time.perf_counter()
            dets, hm_np, *_ = infer(sess, in_name, frame,
                                    args.input_h, args.input_w,
                                    args.conf, args.iou)
            elapsed    = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / (elapsed + 1e-6))

            vis = draw_detections(frame, dets)
            draw_hud(vis, len(dets), fps=fps_smooth, frame_id=frame_id)

            if writer is not None:
                writer.write(vis)

            cv2.imshow("HybridECDet — webcam  (q/ESC quit  s snapshot)", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):   # q or ESC
                break
            if key == ord('s'):
                snap_name = f"snapshot_{snap_idx:04d}.jpg"
                cv2.imwrite(snap_name, vis)
                print(f"[snap]   saved {snap_name}")
                snap_idx += 1

            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"[done]   {frame_id} frames  avg={fps_smooth:.1f} fps")
    if save_path:
        print(f"[saved]  {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='HybridECDet ONNX inference + visualization')
    p.add_argument('--model',      required=True,  help='Path to .onnx file')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',  help='Input image path (image mode)')
    src.add_argument('--video',  help='Input video path (video mode)')
    src.add_argument('--webcam', metavar='CAM_ID', default=None,
                     help='Webcam device id, e.g. 0 (webcam mode)')

    p.add_argument('--output',     default='result.jpg',
                   help='Output path  (image: result.jpg | video: result.mp4)')
    p.add_argument('--output_hm',  default='heatmap.jpg',
                   help='Heatmap overlay output (image mode only)')
    p.add_argument('--input_w',    type=int,   default=832)
    p.add_argument('--input_h',    type=int,   default=512)
    p.add_argument('--num_classes', type=int,  default=10)
    p.add_argument('--conf',       type=float, default=0.3,  help='Confidence threshold')
    p.add_argument('--iou',        type=float, default=0.45, help='NMS IoU threshold')
    p.add_argument('--gpu',        action='store_true', help='Use CUDAExecutionProvider')
    p.add_argument('--no_heatmap',  action='store_true', help='Skip heatmap (image mode only)')
    p.add_argument('--save_webcam', action='store_true',
                   help='Record webcam session to --output file (default: webcam_out.mp4)')
    return p.parse_args()


def main():
    args = parse_args()

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.gpu else ['CPUExecutionProvider'])
    sess    = ort.InferenceSession(args.model, providers=providers)
    in_name = sess.get_inputs()[0].name
    print(f"[onnx]   {args.model}  input='{in_name}'  providers={sess.get_providers()}")

    if args.image:
        run_image(args, sess, in_name)
    elif args.video:
        run_video(args, sess, in_name)
    else:
        run_webcam(args, sess, in_name)


if __name__ == '__main__':
    main()
