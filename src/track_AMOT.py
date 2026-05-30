"""
DEIMv2-DINOv3-S MOT tracking evaluation on VisDrone test-dev sequences.

The model is loaded directly via DEIMv2's YAMLConfig, so the architecture
is 100% config-driven — no separate create_model() wrapper needed.

Reports both:
  - MOT metrics: MOTA / MOTP / IDF1 via MCJDETracker (Kalman + cosine ReID)
  - Detection metrics: COCO-style mAP via VisDrone GT annotations

Usage:
  python src/track_AMOT.py \\
      --deim_config src/configs/deimv2_dinov3_s_visdrone.yml \\
      --load_model  outputs/deimv2_dinov3_s_visdrone/best_stg2.pth \\
      --data_dir    /path/to/UAVdata \\
      --conf_thres  0.4 \\
      [--benchmark]   # pure FPS measurement, then exit
"""
from __future__ import annotations

import _init_paths
import os
import os.path as osp
import sys
import cv2
import logging
import motmetrics as mm
import numpy as np
import torch

from collections import defaultdict
from tqdm import tqdm
from lib.tracker.multitracker import MCJDETracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
import lib.datasets.dataset.jde as datasets

from lib.tracking_utils.utils import mkdir_if_missing


# ── Engine path setup (so 'import engine' resolves our extended engine) ────────

_MODELS = osp.join(osp.dirname(osp.abspath(__file__)), 'lib', 'models')
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import engine  # triggers @register() for all DEIMv2 + MOT extensions


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_model(opt, warmup: int = 30, runs: int = 200):
    """Pure DEIM model inference FPS — no tracker, no Kalman, no matching."""
    from engine.core import YAMLConfig

    cfg   = YAMLConfig(opt.deim_config)
    model = cfg.model
    if opt.load_model:
        state = torch.load(opt.load_model, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model', state), strict=False)
    model = model.to(opt.device).eval()

    W, H  = opt.input_wh
    dummy = torch.zeros(1, 3, H, W, device=opt.device)
    use_cuda = torch.device(opt.device).type == 'cuda'

    print('\n' + '─' * 60)
    print('  Benchmark: DEIMv2-DINOv3-S (DEIM with ReID head)')
    print(f'  config={osp.basename(opt.deim_config)}  input={opt.input_wh}  device={opt.device}')
    print('─' * 60)

    for half in ([False, True] if use_cuda else [False]):
        m = model.half() if half else model
        d = dummy.half() if half else dummy
        dtype_str = 'FP16' if half else 'FP32'
        with torch.no_grad():
            for _ in range(warmup):
                _ = m(d)
            if use_cuda:
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for _ in range(runs):
                    _ = m(d)
                t1.record()
                torch.cuda.synchronize()
                ms = t0.elapsed_time(t1) / runs
            else:
                import time
                s = time.perf_counter()
                for _ in range(runs):
                    _ = m(d)
                ms = (time.perf_counter() - s) * 1000 / runs
        print(f'  {dtype_str}  latency: {ms:.1f} ms   FPS: {1000/ms:.1f}')
    print('─' * 60 + '\n')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tlwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] += out[:, 0]
    out[:, 3] += out[:, 1]
    return out


def load_gt_detections(ann_file: str) -> tuple:
    """
    Read VisDrone annotation file.
    Returns (frame_dets, frame_ignores) dicts:
      frame_dets   : {fid: [(tlwh, cls_id_0idx), ...]}
      frame_ignores: {fid: [tlwh, ...]}
    """
    frame_dets:    dict = {}
    frame_ignores: dict = {}
    if not osp.isfile(ann_file):
        return frame_dets, frame_ignores

    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            fid    = int(parts[0])
            tlwh   = np.array([float(x) for x in parts[2:6]], dtype=np.float32)
            score  = int(float(parts[6]))
            cat    = int(float(parts[7]))
            cls_id = cat - 1

            if score == 0 or cat == 0 or cat == 11:
                frame_ignores.setdefault(fid, []).append(tlwh)
                continue
            if cls_id < 0:
                continue
            frame_dets.setdefault(fid, []).append((tlwh, cls_id))

    return frame_dets, frame_ignores


# ── Core tracking loop ────────────────────────────────────────────────────────

def write_results_dict(file_name, results_dict, num_classes=10):
    _skip_cls = {1, 2, 6, 7, 9}
    save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},-1,-1\n'
    with open(file_name, 'w') as f:
        for cls_id in range(num_classes):
            if cls_id in _skip_cls:
                continue
            for frame_id, tlwhs, track_ids, scores in results_dict[cls_id]:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    f.write(save_format.format(
                        frame=frame_id, id=track_id,
                        x1=x1, y1=y1, w=w, h=h,
                        score=score, cls_id=cls_id + 1))
    logger.info(f'saved results → {file_name}')


def eval_seq(opt, data_loader, result_f_name,
             save_dir=None, show_image=False, frame_rate=30):
    """
    Run DEIMv2 tracker on one sequence.
    Returns (n_frames, avg_time, n_calls, frame_det_results).
    """
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = MCJDETracker(opt, frame_rate)
    timer   = Timer()
    results_dict     = defaultdict(list)
    frame_det_results: dict = {}
    frame_id = 0

    pbar = tqdm(data_loader, desc='Tracking', unit='frame', dynamic_ncols=True)
    for path, img, img0 in pbar:
        frame_id += 1
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)
        timer.tic()
        online_targets_dict = tracker.update_tracking(blob, img0)
        timer.toc()

        fps = 1.0 / max(1e-5, timer.average_time)
        pbar.set_postfix(fps=f'{fps:.1f}')

        online_tlwhs_dict  = defaultdict(list)
        online_ids_dict    = defaultdict(list)
        online_scores_dict = defaultdict(list)

        for cls_id in range(opt.num_classes):
            for track in online_targets_dict[cls_id]:
                tlwh  = track.curr_tlwh
                t_id  = track.track_id
                score = track.score
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs_dict[cls_id].append(tlwh)
                    online_ids_dict[cls_id].append(t_id)
                    online_scores_dict[cls_id].append(score)

        # Raw DETR detections for mAP
        frame_dets_this = []
        if hasattr(tracker, 'last_raw_dets'):
            for cls_id, raw in tracker.last_raw_dets.items():
                for det in raw:
                    frame_dets_this.append((det[:4].copy(), float(det[4]), int(cls_id)))
        frame_det_results[frame_id] = frame_dets_this

        for cls_id in range(opt.num_classes):
            results_dict[cls_id].append((frame_id,
                                         online_tlwhs_dict[cls_id],
                                         online_ids_dict[cls_id],
                                         online_scores_dict[cls_id]))

        if show_image or save_dir is not None:
            online_im = vis.plot_tracks(
                image=img0, tlwhs_dict=online_tlwhs_dict,
                obj_ids_dict=online_ids_dict, num_classes=opt.num_classes,
                scores=online_scores_dict, frame_id=frame_id,
                fps=1.0 / timer.average_time)
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(osp.join(save_dir, f'{frame_id:05d}.jpg'), online_im)

    write_results_dict(result_f_name, results_dict, opt.num_classes)
    return frame_id, timer.average_time, timer.calls, frame_det_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(opt, data_root='', seqs=('',), exp_name='',
         save_images=False, show_image=False):
    logger.setLevel(logging.INFO)
    result_root = osp.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)

    accs = []
    timer_avgs, timer_calls = [], []
    total_frames = 0

    det_ev = COCOEvaluator(
        num_classes=opt.num_classes,
        class_names=VISDRONE_CLASSES[:opt.num_classes],
    )

    for seq in seqs:
        output_dir = (osp.join(data_root, '..', 'outputs', exp_name, seq)
                      if save_images else None)
        logger.info(f'start seq: {seq}')

        dataloader = datasets.LoadImages(
            osp.join(data_root, seq),
            opt.img_size,
            use_imagenet_norm=True,   # DEIMv2 DINOv3 always uses ImageNet norm
        )
        result_filename = osp.join(result_root, f'{seq}.txt')

        nf, ta, tc, frame_det_results = eval_seq(
            opt, dataloader, result_filename,
            save_dir=output_dir, show_image=show_image, frame_rate=30,
        )
        total_frames  += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # Detection mAP
        ann_root = osp.join(osp.dirname(data_root), 'annotations')
        ann_file = osp.join(ann_root, f'{seq}.txt')
        gt_frame_dets, gt_frame_ignores = load_gt_detections(ann_file)

        for fid in tqdm(sorted(set(frame_det_results) | set(gt_frame_dets)),
                        desc=f'  mAP [{seq}]', unit='frame', leave=False):
            preds   = frame_det_results.get(fid, [])
            gts     = gt_frame_dets.get(fid, [])
            ignores = gt_frame_ignores.get(fid, [])

            if preds:
                pb, ps, pl = zip(*preds)
                pred_boxes  = np.stack(pb).astype(np.float32)
                pred_scores = np.array(ps, dtype=np.float32)
                pred_labels = np.array(pl, dtype=np.int64)
            else:
                pred_boxes  = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros((0,),   dtype=np.float32)
                pred_labels = np.zeros((0,),   dtype=np.int64)

            if gts:
                gb, gl = zip(*gts)
                gt_boxes  = _tlwh_to_xyxy(np.stack(gb))
                gt_labels = np.array(gl, dtype=np.int64)
            else:
                gt_boxes  = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.zeros((0,),   dtype=np.int64)

            ignore_boxes = _tlwh_to_xyxy(np.stack(ignores)) if ignores else None
            det_ev.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, ignore_boxes)

        # MOT tracking metrics
        logger.info(f'Evaluate seq: {seq}')
        evaluator = Evaluator(ann_root, seq, 'mot')
        accs.append(evaluator.eval_file(result_filename))

    # Speed
    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time    = np.dot(timer_avgs, timer_calls)
    avg_time    = all_time / np.sum(timer_calls)
    logger.info(f'Frames: {total_frames}  |  Time: {all_time:.2f}s  |  FPS: {1.0/avg_time:.2f}')

    # Tracking metrics
    metrics = mm.metrics.motchallenge_metrics
    mh      = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print('\n===== Tracking Metrics =====')
    print(strsummary)
    Evaluator.save_summary(summary, osp.join(result_root, f'summary_{exp_name}.xlsx'))

    # Detection metrics
    print('\n===== Detection Metrics =====')
    det_stats = det_ev.summarize()
    det_ev.print_summary(det_stats)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DEIMv2 MOT tracking evaluation')
    parser.add_argument('--deim_config',  type=str, required=True,
                        help='Path to YAML config, e.g. src/configs/deimv2_dinov3_s_visdrone.yml')
    parser.add_argument('--load_model',   type=str, default='',
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--data_dir',     type=str, default='/media/jianbo/ioe/UAVdata')
    parser.add_argument('--gpus',         type=str, default='0')
    parser.add_argument('--num_classes',  type=int, default=10)
    parser.add_argument('--conf_thres',   type=float, default=0.4)
    parser.add_argument('--nms_thres',    type=float, default=0.4)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--min-box-area', type=float, default=100, dest='min_box_area')
    parser.add_argument('--input-wh', type=lambda s: tuple(int(x) for x in s.split(',')),
                        default=(1280, 704), dest='input_wh')
    parser.add_argument('--save_dir_result', type=str, default='deimv2_dinov3_s_visdrone')
    parser.add_argument('--save_images', action='store_true', default=False)
    parser.add_argument('--test_visdrone', type=bool, default=True)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--reid_cls_ids', type=str, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--K', type=int, default=500,
                        help='max queries (should match num_queries in YAML)')

    opt = parser.parse_args()

    # Derived fields
    gpus = [int(g) for g in opt.gpus.split(',')]
    opt.device = f'cuda:{gpus[0]}' if gpus[0] >= 0 else 'cpu'
    W, H = opt.input_wh
    opt.img_size   = (W, H)
    opt.heads      = {}
    opt.head_conv  = 256
    opt.down_ratio = 16   # S16 stride for reid_motion helpers

    if opt.benchmark:
        benchmark_model(opt)
        sys.exit(0)

    seqs_str = ''
    if opt.test_visdrone:
        seqs_str = '''uav0000009_03358_v
                      uav0000073_00600_v
                      uav0000073_04464_v
                      uav0000077_00720_v
                      uav0000088_00290_v
                      uav0000119_02301_v
                      uav0000120_04775_v
                      uav0000161_00000_v
                      uav0000188_00000_v
                      uav0000201_00000_v
                      uav0000249_00001_v
                      uav0000249_02688_v
                      uav0000297_00000_v
                      uav0000297_02761_v
                      uav0000306_00230_v
                      uav0000355_00001_v
                      uav0000370_00001_v
                      '''

    data_root = osp.join(opt.data_dir, 'VisDrone2019/test_dev/sequences')
    seqs      = [s.strip() for s in seqs_str.split()]
    exp_name  = opt.save_dir_result

    print(f'Results → {exp_name}  |  device: {opt.device}')
    main(opt, data_root=data_root, seqs=seqs, exp_name=exp_name,
         save_images=opt.save_images, show_image=False)
