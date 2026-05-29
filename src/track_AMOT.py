from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core._multiarray_umath import ndarray
import _init_paths
import os
import os.path as osp
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
from lib.utils.det_eval import DetectionEvaluator
import lib.datasets.dataset.jde as datasets

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_model(opt, warmup: int = 30, runs: int = 200):
    """
    Measure pure model inference FPS — no tracker, no Kalman, no matching.
    Prints latency and FPS for FP32 and (optionally) FP16.
    """
    from lib.models.model import create_model, load_model

    print('\n' + '─' * 55)
    print('  Benchmark: DEIMMotNet pure inference')
    print(f'  arch={opt.arch}  input={opt.input_wh}  device={opt.device}')
    print('─' * 55)

    model = create_model(opt.arch, opt.heads, opt.head_conv,
                         num_classes=opt.num_classes, opt=opt)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device).eval()

    W, H  = opt.input_wh                              # (W, H) tuple
    dummy = torch.zeros(1, 3, H, W, device=opt.device)

    device = torch.device(opt.device)
    use_cuda = device.type == 'cuda'

    for half in ([False, True] if use_cuda else [False]):
        m = model.half() if half else model
        d = dummy.half() if half else dummy
        dtype_str = 'FP16' if half else 'FP32'

        with torch.no_grad():
            # warmup
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

    print('─' * 55 + '\n')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tlwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) tlwh → xyxy"""
    out = boxes.copy()
    out[:, 2] += out[:, 0]
    out[:, 3] += out[:, 1]
    return out


def load_gt_detections(ann_file: str) -> tuple:
    """
    Read VisDrone annotation file.

    Format: frame_id, target_id, x, y, w, h, score, object_category, truncation, occlusion
      score=1 → valid,  score=0 → ignored/don't-care
      object_category: 1-indexed (1=pedestrian … 10=motor, 0=ignored-region, 11=others)

    Returns:
        frame_dets   : {frame_id: [(tlwh, cls_id_0idx), ...]}  — valid objects only
        frame_ignores: {frame_id: [tlwh, ...]}                  — ignored regions
    """
    frame_dets:    dict = {}
    frame_ignores: dict = {}
    if not osp.isfile(ann_file):
        return frame_dets, frame_ignores

    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            fid    = int(parts[0])
            tlwh   = np.array([float(x) for x in parts[2:6]], dtype=np.float32)
            score  = int(float(parts[6]))      # 1=valid, 0=ignored
            cat    = int(float(parts[7]))      # 1-indexed category
            cls_id = cat - 1                   # 0-indexed

            # ignored region or invalid object → don't-care
            if score == 0 or cat == 0 or cat == 11:
                frame_ignores.setdefault(fid, []).append(tlwh)
                continue

            if cls_id < 0:
                continue
            frame_dets.setdefault(fid, []).append((tlwh, cls_id))

    return frame_dets, frame_ignores


# ── Core tracking / detection loop ────────────────────────────────────────────

def write_results_dict(file_name, results_dict, data_type, num_classes=10):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score}, {cls_id},-1, -1\n'
    else:
        raise ValueError(data_type)
    with open(file_name, 'w') as f:
        for cls_id in range(num_classes):
            if cls_id == 1 or cls_id == 2 or cls_id == 6 or cls_id == 7 or cls_id == 9:
                continue
            cls_results = results_dict[cls_id]
            for frame_id, tlwhs, track_ids, scores in cls_results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    re_cls_id = cls_id + 1
                    line = save_format.format(frame=frame_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              score=score,
                                              cls_id=re_cls_id)
                    f.write(line)
    logger.info('save results to {}'.format(file_name))


def eval_seq(opt,
             data_loader,
             data_type,
             result_f_name,
             save_dir=None,
             show_image=True,
             frame_rate=30):
    """
    Run tracker on one sequence.

    Returns:
        (n_frames, avg_time, n_calls, frame_det_results)

        frame_det_results : {frame_id: list of (xyxy, score, cls_id)}
            — raw detector outputs at score>=0.01 for mAP computation,
              independent of tracking conf_thres
    """
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = MCJDETracker(opt, frame_rate)

    timer = Timer()
    results_dict = defaultdict(list)
    frame_det_results: dict = {}   # {frame_id: [(xyxy, score, cls_id)]}
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
            online_targets = online_targets_dict[cls_id]
            for track in online_targets:
                tlwh  = track.curr_tlwh
                t_id  = track.track_id
                score = track.score
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs_dict[cls_id].append(tlwh)
                    online_ids_dict[cls_id].append(t_id)
                    online_scores_dict[cls_id].append(score)

        # Collect raw detector outputs for mAP (score >= 0.01, no tracking filters).
        # tracker.last_raw_dets contains xyxy boxes — no conversion needed.
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
            if frame_id > 0:
                online_im: ndarray = vis.plot_tracks(image=img0,
                                                     tlwhs_dict=online_tlwhs_dict,
                                                     obj_ids_dict=online_ids_dict,
                                                     num_classes=opt.num_classes,
                                                     scores=online_scores_dict,
                                                     frame_id=frame_id,
                                                     fps=1.0 / timer.average_time)
        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

    write_results_dict(result_f_name, results_dict, data_type)
    return frame_id, timer.average_time, timer.calls, frame_det_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(opt,
         data_root='',
         seqs=('',),
         exp_name='',
         save_images=False,
         save_videos=False,
         show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    use_imagenet_norm = 'deim' in opt.arch or 'hybrid' in opt.arch

    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
    det_ev = COCOEvaluator(num_classes=opt.num_classes,
                           class_names=VISDRONE_CLASSES[:opt.num_classes])

    for seq in seqs:
        output_dir = os.path.join(
            data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))

        dataloader = datasets.LoadImages(
            osp.join(data_root, seq), opt.img_size,
            use_imagenet_norm=use_imagenet_norm)

        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        frame_rate = 30

        nf, ta, tc, frame_det_results = eval_seq(
            opt, dataloader, data_type, result_filename,
            save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)

        n_frame       += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # ── Detection mAP: compare predictions vs GT per frame ───────────────
        ann_root = osp.join(osp.dirname(data_root), 'annotations')
        ann_file = osp.join(ann_root, f'{seq}.txt')
        gt_frame_dets, gt_frame_ignores = load_gt_detections(ann_file)

        all_frame_ids = sorted(set(frame_det_results.keys()) | set(gt_frame_dets.keys()))
        for fid in tqdm(all_frame_ids, desc=f'  mAP [{seq}]', unit='frame',
                        dynamic_ncols=True, leave=False):
            preds   = frame_det_results.get(fid, [])
            gts     = gt_frame_dets.get(fid, [])
            ignores = gt_frame_ignores.get(fid, [])

            if preds:
                pb, ps, pl = zip(*preds)
                # pred_boxes are already xyxy from last_raw_dets
                pred_boxes  = np.stack(pb).astype(np.float32)
                pred_scores = np.array(ps, dtype=np.float32)
                pred_labels = np.array(pl, dtype=np.int64)
            else:
                pred_boxes  = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros((0,),   dtype=np.float32)
                pred_labels = np.zeros((0,),   dtype=np.int64)

            if gts:
                gb_list, gl_list = zip(*gts)
                gt_boxes  = _tlwh_to_xyxy(np.stack(gb_list))
                gt_labels = np.array(gl_list, dtype=np.int64)
            else:
                gt_boxes  = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.zeros((0,),   dtype=np.int64)

            ignore_boxes = (
                _tlwh_to_xyxy(np.stack(ignores)) if ignores else None
            )

            det_ev.update(pred_boxes, pred_scores, pred_labels,
                          gt_boxes,  gt_labels, ignore_boxes)

        # ── MOT tracking metrics ─────────────────────────────────────────────
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(ann_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # ── Speed ─────────────────────────────────────────────────────────────────
    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
        all_time, 1.0 / avg_time))

    # ── Tracking metrics (MOTA, MOTP, IDF1, …) ───────────────────────────────
    metrics = mm.metrics.motchallenge_metrics
    mh      = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print('\n===== Tracking Metrics =====')
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(
        result_root, 'summary_{}.xlsx'.format(exp_name)))

    # ── Detection metrics (COCO-style) ───────────────────────────────────────
    print('\n===== Detection Metrics =====')
    det_stats = det_ev.summarize()
    det_ev.print_summary(det_stats)


if __name__ == '__main__':
    opt = opts().init()

    # ── Pure inference benchmark (no tracking) ────────────────────────────────
    import sys as _sys
    if '--benchmark' in _sys.argv:
        opt.device = getattr(opt, 'device', 'cuda:0')
        benchmark_model(opt)
        _sys.exit(0)

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
    data_root = os.path.join(opt.data_dir, 'VisDrone2019/test_dev/sequences')
    seqs = [seq.strip() for seq in seqs_str.split()]
    print(opt.save_dir_result)
    opt.device = 'cuda:0'
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='save_name',
         show_image=False,
         save_images=True,
         save_videos=False)
