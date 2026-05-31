from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--task', default='mot', help='mot')
        self.parser.add_argument('--dataset', default='jde', help='jde')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume training; reloads optimizer and sets '
                                      'load_model to model_last.pth if load_model is empty')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, comma-separated for multiple GPUs')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='dataloader threads')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable cudnn benchmark (use when input size varies)')
        self.parser.add_argument('--seed', type=int, default=317)
        self.parser.add_argument('--is_debug', type=bool, default=False,
                                 help='single-process debug mode')

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='print to screen instead of progress bar')
        self.parser.add_argument('--hide_data_time', action='store_true')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save checkpoint every epoch')
        self.parser.add_argument('--metric', default='loss')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5)

        # model
        self.parser.add_argument('--arch', default='ecdet_jde',
                                 help='ecdet_jde (ECViT + HybridEncoder + ECTransformer)')

        # ECDetJDE — paths and runtime settings only
        # Architecture params (nhead, dim_ff, hidden_dim, num_layers, etc.) are
        # hardcoded per variant in models/ecdet_jde/model.py::_ECVIT_CONFIGS.
        self.parser.add_argument('--ecvit_name', default='ecvitt',
                                 help='variant: ecvitt | ecvittplus | ecvits | ecvitsplus')
        self.parser.add_argument('--ecvit_weights', default='',
                                 help='path to ECViT backbone weights (.pth)')
        self.parser.add_argument('--ecdet_pretrained', default='',
                                 help='path to full ECDet COCO checkpoint (.pth)')
        self.parser.add_argument('--eval_spatial_size', type=int, nargs=2,
                                 default=[608, 1088], help='[H, W] for anchor pre-generation')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='kept for hm GT generation in dataset')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1)
        self.parser.add_argument('--input_h',   type=int, default=-1)
        self.parser.add_argument('--input_w',   type=int, default=-1)
        self.parser.add_argument('--input-wh', type=int, nargs=2, default=[1088, 608],
                                 help='default net input resolution W H (e.g. 1088 608)')

        # train
        self.parser.add_argument('--lr', type=float, default=7e-4)
        self.parser.add_argument('--lr_step', type=str, default='10,20',
                                 help='epochs to drop LR by 10x')
        self.parser.add_argument('--num_epochs', type=int, default=30)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--master_batch_size', type=int, default=-1)
        self.parser.add_argument('--num_iters', type=int, default=-1)
        self.parser.add_argument('--val_intervals', type=int, default=5)
        self.parser.add_argument('--trainval', action='store_true')

        # inference / test
        self.parser.add_argument('--K', type=int, default=200,
                                 help='max detections per image')
        self.parser.add_argument('--fix_res', action='store_true')
        self.parser.add_argument('--keep_res', action='store_true')

        # tracking
        self.parser.add_argument('--test_visdrone', default=True)
        self.parser.add_argument('--test_uavdt',    default=False)
        self.parser.add_argument('--conf_thres', type=float, default=0.4,
                                 help='detection confidence threshold')
        self.parser.add_argument('--det_thres',  type=float, default=0.3)
        self.parser.add_argument('--nms_thres',  type=float, default=0.4)
        self.parser.add_argument('--track_buffer', type=int, default=30)
        self.parser.add_argument('--min-box-area', type=float, default=100,
                                 help='filter out boxes smaller than this area')

        # I/O
        self.parser.add_argument('--input-mode', type=str, default='video',
                                 help='video | image_dir')
        self.parser.add_argument('--input-video', type=str, default='')
        self.parser.add_argument('--input-img',   type=str, default='')
        self.parser.add_argument('--output-format', type=str, default='video')
        self.parser.add_argument('--output-root',   type=str, default='../results')
        self.parser.add_argument('--save_dir_result', type=str, default='results')

        # dataset
        self.parser.add_argument('--data_cfg', type=str,
                                 default='../src/lib/cfg/visdrone.json')
        self.parser.add_argument('--data_dir', type=str, default='')

        # loss
        self.parser.add_argument('--id_weight', type=float, default=1.0,
                                 help='weight for ReID loss (0 = detection only)')
        self.parser.add_argument('--tri', action='store_true',
                                 help='add triplet loss to ReID')

        # ReID
        self.parser.add_argument('--reid_dim', type=int, default=128,
                                 help='ReID embedding dimension')
        self.parser.add_argument('--reid_cls_ids', default='0,1,2,3,4,5,6,7,8,9',
                                 help='class IDs that need ReID')

        # distributed
        self.parser.add_argument('--local-rank', type=int, default=0)

    def parse(self, args=''):
        opt = self.parser.parse_args() if args == '' else self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus     = [int(g) for g in opt.gpus.split(',')]
        opt.lr_step  = [int(s) for s in opt.lr_step.split(',')]
        opt.fix_res  = not opt.keep_res

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest = opt.batch_size - opt.master_batch_size
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            chunk = rest // (len(opt.gpus) - 1)
            if i < rest % (len(opt.gpus) - 1):
                chunk += 1
            opt.chunk_sizes.append(chunk)
        print('chunk_sizes:', opt.chunk_sizes)

        opt.root_dir  = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.exp_dir   = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir  = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('Output will be saved to', opt.save_dir)

        if opt.resume and opt.load_model == '':
            base = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
            opt.load_model = os.path.join(base, 'model_last.pth')

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_input_wh
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes   = dataset.num_classes
        print('num_classes:', opt.num_classes)

        for reid_id in opt.reid_cls_ids.split(','):
            if int(reid_id) > opt.num_classes - 1:
                print('[Err]: reid_cls_ids conflicts with num_classes')
                return

        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h  = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w  = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res  = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'mot':
            if opt.id_weight > 0:
                opt.nID_dict = dataset.nID_dict
        else:
            assert 0, 'task not defined!'

        return opt

    def init(self, args=''):
        opt = self.parse(args)

        default_dataset_info = {
            'mot': {
                'default_input_wh': [opt.input_wh[1], opt.input_wh[0]],
                'num_classes': len(opt.reid_cls_ids.split(',')),
                'mean': [0.408, 0.447, 0.470],
                'std':  [0.289, 0.274, 0.278],
                'dataset': 'jde',
                'nID_dict': {},
            },
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    setattr(self, k, v)

        h_w = default_dataset_info[opt.task]['default_input_wh']
        opt.img_size = (h_w[1], h_w[0])
        print('Net input: {:d}×{:d}'.format(h_w[1], h_w[0]))

        dataset    = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
