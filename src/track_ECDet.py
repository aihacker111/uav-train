"""
track_ECDet.py — clean ByteTrack-style tracking with ECDetJDE.

Replaces track_AMOT.py / MCJDETracker with MCByteTracker which eliminates
ghost boxes by removing the update_retrack propagation path and properly
using the dual-threshold secondary detection pool from ECDetJDEPostProcessor.
"""

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
from lib.tracker.multitracker import MCByteTracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
import lib.datasets.dataset.jde as datasets

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts


def write_results_dict(file_name, results_dict, data_type, num_classes=10):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},-1,-1\n'
    else:
        raise ValueError(data_type)
    with open(file_name, 'w') as f:
        for cls_id in range(num_classes):
            if cls_id in (1, 2, 6, 7, 9):
                continue
            cls_results = results_dict[cls_id]
            for frame_id, tlwhs, track_ids, scores in cls_results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    re_cls_id = cls_id + 1
                    line = save_format.format(
                        frame=frame_id, id=track_id,
                        x1=x1, y1=y1, w=w, h=h,
                        score=score, cls_id=re_cls_id)
                    f.write(line)
    logger.info('save results to {}'.format(file_name))


def eval_seq(opt, data_loader, data_type, result_f_name,
             save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = MCByteTracker(opt, frame_rate)
    timer   = Timer()
    results_dict = defaultdict(list)
    frame_id = 0

    for path, img, img0 in data_loader:
        if frame_id % 30 == 0 and frame_id != 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1.0 / max(1e-5, timer.average_time)))
        frame_id += 1

        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)
        timer.tic()
        online_targets_dict = tracker.update_tracking(blob, img0)
        timer.toc()

        online_tlwhs_dict  = defaultdict(list)
        online_ids_dict    = defaultdict(list)
        online_scores_dict = defaultdict(list)
        max_area = img0.shape[0] * img0.shape[1] * 0.40

        for cls_id in range(opt.num_classes):
            for track in online_targets_dict[cls_id]:
                tlwh     = track.curr_tlwh
                t_id     = track.track_id
                score    = track.score
                box_area = tlwh[2] * tlwh[3]
                if opt.min_box_area < box_area < max_area:
                    online_tlwhs_dict[cls_id].append(tlwh)
                    online_ids_dict[cls_id].append(t_id)
                    online_scores_dict[cls_id].append(score)

        for cls_id in range(opt.num_classes):
            results_dict[cls_id].append((
                frame_id,
                online_tlwhs_dict[cls_id],
                online_ids_dict[cls_id],
                online_scores_dict[cls_id],
            ))

        if show_image or save_dir is not None:
            if frame_id > 0:
                online_im: ndarray = vis.plot_tracks(
                    image=img0,
                    tlwhs_dict=online_tlwhs_dict,
                    obj_ids_dict=online_ids_dict,
                    num_classes=opt.num_classes,
                    scores=online_scores_dict,
                    frame_id=frame_id,
                    fps=1.0 / timer.average_time,
                )

        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

    write_results_dict(result_f_name, results_dict, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='', seqs=('',), exp_name='',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    for seq in seqs:
        output_dir = os.path.join(
            data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        frame_rate = 30
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                               save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    timer_avgs  = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time    = np.dot(timer_avgs, timer_calls)
    avg_time    = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    metrics   = mm.metrics.motchallenge_metrics
    mh        = mm.metrics.create()
    summary   = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

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
