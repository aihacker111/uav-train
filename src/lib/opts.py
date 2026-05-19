from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--task', default='mot', help='mot | hybrid')
        self.parser.add_argument('--dataset', default='jde', help='jde')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--load_model',
                                 default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume',
                                 action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus',
                                 default='0',  # 0, 5, 6
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=8,  # 8, 6, 4
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet
        self.parser.add_argument('--gen-scale',
                                 action='store_true',
                                 default=True,
                                 help='Whether to generate multi-scales')
        self.parser.add_argument('--no-gen-scale',
                                 action='store_false',
                                 dest='gen_scale',
                                 help='Disable multi-scale generation')
        self.parser.add_argument('--is_debug',
                                 action='store_true',
                                 help='Enable debug mode (forces single-process data loading)')

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                                 help='visualization threshold.')

        # model: backbone and so on...
        self.parser.add_argument('--arch',
                                 default='lwdetr_tiny',
                                 help='model architecture. Currently supported: '
                                      'lwdetr_tiny | lwdetr_small | lwdetr_base | '
                                      'hybrid_tiny | hybrid_small | hybrid_base')
        self.parser.add_argument('--head_conv',
                                 type=int,
                                 default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: 256.')
        self.parser.add_argument('--backbone_lr_scale',
                                 type=float,
                                 default=0.2,
                                 help='LR multiplier for ViT backbone (relative to head LR). '
                                      '0.2 allows faster UAV-domain adaptation than 0.1 '
                                      'while still protecting pretrained features.')
        self.parser.add_argument('--freeze_backbone_epochs',
                                 type=int,
                                 default=0,
                                 help='Number of epochs to freeze ViT backbone before fine-tuning. 0 = no freeze.')
        self.parser.add_argument('--backbone_weights',
                                 type=str,
                                 default='',
                                 help='Path to LW-DETR COCO pretrained checkpoint (.pth). '
                                      'Empty string = train from scratch.')
        self.parser.add_argument('--grad_checkpoint',
                                 action='store_true',
                                 default=False,
                                 help='Enable gradient checkpointing on the ViT backbone. '
                                      'Reduces backbone VRAM by ~50%% to allow larger batch '
                                      'sizes on memory-constrained GPUs. Cost: ~20%% slower backward.')
        self.parser.add_argument('--num_output_levels',
                                 type=int,
                                 default=1,
                                 help='Number of feature pyramid levels emitted by MultiScaleNeck '
                                      'to the DETR decoder (1=single-scale P4, 2=P4+P5, 3=P4+P5+P6). '
                                      'Level 1 is pretrained-compatible. Level >1 changes decoder '
                                      'attention shape so cross-attention restarts from scratch.')
        self.parser.add_argument('--top_down_fusion',
                                 action='store_true',
                                 default=False,
                                 help='Enable FPN top-down fusion in MultiScaleNeck. '
                                      'REQUIRES --num_output_levels > 1 to have any effect. '
                                      'Adds a lateral 1x1 conv from each coarser level back to '
                                      'the finer level, giving finer features global context.')
        self.parser.add_argument('--down_ratio',
                                 type=int,
                                 default=4,  # 输出特征图的下采样率 H=H_image/4 and W=W_image/4
                                 help='output stride. Currently only supports 4.')
        
        self.parser.add_argument('--seg_feat_channel', default=8, type=int, help='.')
        

        # input
        self.parser.add_argument('--input_res',
                                 type=int,
                                 default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h',
                                 type=int,
                                 default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w',
                                 type=int,
                                 default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=4e-4,
                                 help='learning rate for batch size 16 (single GPU). '
                                      'Linear-scale with batch: 8e-4 for bs=32, 2e-4 for bs=8.')
        self.parser.add_argument('--lr_scale',
                                 type=str,
                                 default='linear',
                                 choices=['linear', 'sqrt', 'none'],
                                 help='Multi-GPU LR scaling rule. '
                                      'linear: lr *= (gpus * batch_size) / base_batch_size  '
                                      'sqrt:   lr *= sqrt(gpus * batch_size / base_batch_size)  '
                                      'none:   no scaling (manual tuning).')
        self.parser.add_argument('--base_batch_size',
                                 type=int,
                                 default=16,
                                 help='Reference batch size the base --lr was tuned for '
                                      '(single GPU). Match this to your single-GPU --batch_size. '
                                      'Used by --lr_scale for multi-GPU LR adjustment.')
        self.parser.add_argument('--lr_step',
                                 type=str,
                                 default='10, 20',
                                 help='drop learning rate by 10.')

        self.parser.add_argument('--cosine_lr',
                                 default=True,
                                 action=argparse.BooleanOptionalAction,
                                 help='Use cosine LR schedule with linear warmup instead of step decay. '
                                      'Enabled by default (best practice for ViT). Use --no-cosine-lr to disable.')
        self.parser.add_argument('--warmup_iters',
                                 type=int,
                                 default=500,
                                 help='Linear LR warmup iterations (cosine_lr only). '
                                      '500 ≈ 0.8 epoch at batch=16, ~10k samples.')
        self.parser.add_argument('--min_lr_ratio',
                                 type=float,
                                 default=0.01,
                                 help='Minimum LR as fraction of base LR at end of cosine decay.')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=30,  # 30, 10, 3, 1
                                 help='total training epochs.')
        self.parser.add_argument('--close_mosaic_epochs',
                                 type=int,
                                 default=10,
                                 help='Disable Mosaic/MixUp/Perspective in the last N epochs '
                                      '(YOLOv5/v8 close-mosaic trick). 0 = never disable.')
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=4,  # 1920×1088: 4 per GPU (24GB); 1088×640: 8-12 per GPU
                                 help='batch size per GPU. '
                                      'ZR10 1920×1088: use 4 (24GB GPU) or 2 (16GB GPU). '
                                      'ZR10 1088×640 tracking: use 8-12.')

        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--use_amp',
                                 action='store_true',
                                 default=False,
                                 help='Enable Automatic Mixed Precision (fp16 forward + fp32 '
                                      'weights). Gives ~1.5-2x training speedup and ~50%% VRAM '
                                      'reduction with negligible accuracy loss. Requires CUDA.')
        self.parser.add_argument('--grad_clip',
                                 type=float,
                                 default=0.1,
                                 help='Max gradient norm for gradient clipping. '
                                      '0.1 protects ViT backbone from gradient explosion. '
                                      'Set to 0.0 to disable.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval',
                                 action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument('--K',
                                 type=int,
                                 default=200,  # 128
                                 help='max number of output objects.')  # 一张图输出检测目标最大数量
        self.parser.add_argument('--not_prefetch_test',
                                 action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res',
                                 action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res',
                                 action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        # tracking
        # tracking
        self.parser.add_argument(
            '--test_uavdt', default=False, help='test_visdrone')
        self.parser.add_argument(
            '--test_visdrone', default=True, help='test_visdrone')

        self.parser.add_argument(
            '--conf_thres',
            type=float,
            default=0.4,  # 0.6, 0.4
            help='confidence thresh for tracking')  # heat-map置信度阈值

        self.parser.add_argument('--det_thres',
                                 type=float,
                                 default=0.3,
                                 help='confidence thresh for detection')

        self.parser.add_argument('--nms_thres',
                                 type=float,
                                 default=0.4,
                                 help='iou thresh for nms')

        self.parser.add_argument('--track_buffer',
                                 type=int,
                                 default=30,  # 30
                                 help='tracking buffer')
        self.parser.add_argument('--min-box-area',
                                 type=float,
                                 default=100,
                                 help='filter out tiny boxes')

        # 测试阶段的输入数据模式: video or image dir
        self.parser.add_argument('--input-mode',
                                 type=str,
                                 default='video',  # video or image_dir or img_path_list_txt
                                 help='input data type(video or image dir)')

        # 输入的video文件路径
        self.parser.add_argument('--input-video',
                                 type=str,
                                 default='../videos/uav_339.mp4',
                                 help='path to the input video')

        # 输入的image目录
        self.parser.add_argument('--input-img',
                                 type=str,
                                 default='/users/duanyou/c5/all_pretrain/test.txt',  # ../images/
                                 help='path to the input image directory or image file list(.txt)')

        self.parser.add_argument('--output-format',
                                 type=str,
                                 default='video',
                                 help='video or text')
        self.parser.add_argument('--output-root',
                                 type=str,
                                 default='../results',
                                 help='expected output root path')

        # mot: 选择数据集的配置文件
        self.parser.add_argument('--data_cfg', type=str,
                                 default='../src/lib/cfg/visdrone.json',  # 'mcmot_det.json', 'visdrone.json'
                                 help='load data from cfg')
        # self.parser.add_argument('--data_cfg', type=str,
        #                          default='../src/lib/cfg/mcmot.json',  # mcmot.json, mcmot_det.json,
        #                          help='load data from cfg')
        self.parser.add_argument('--data_dir',
                                 type=str,
                                 default='/media/jianbo/ioe/UAVdata')

        # hybrid model loss weights
        self.parser.add_argument('--bbox_weight', type=float, default=5.0,
                                 help='DETR SmoothL1 box loss weight (hybrid task).')
        self.parser.add_argument('--giou_weight', type=float, default=2.0,
                                 help='DETR CIoU loss weight (hybrid task).')
        self.parser.add_argument('--stage1_weight', type=float, default=1.0,
                                 help='Weight for CenterNet stage-1 loss (hybrid task).')
        self.parser.add_argument('--stage2_weight', type=float, default=1.0,
                                 help='Weight for DETR stage-2 loss (hybrid task).')
        self.parser.add_argument('--consist_weight', type=float, default=0.1,
                                 help='Stage-1/Stage-2 consistency loss weight. '
                                      'Ramped up over consist_warmup_epochs to avoid '
                                      'early-training noise from random Stage-2 matches.')
        self.parser.add_argument('--consist_warmup_epochs', type=int, default=10,
                                 help='Epochs over which consistency loss weight ramps '
                                      'from 0 to consist_weight.')
        self.parser.add_argument('--grad_accum', type=int, default=1,
                                 help='Gradient accumulation steps. Effective batch = '
                                      'batch_size * grad_accum. Use to simulate larger '
                                      'batches on memory-constrained hardware.')

        # loss
        self.parser.add_argument('--mse_loss',  # default: false
                                 action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')
        self.parser.add_argument('--reg_loss',
                                 default='l1',
                                 help='regression loss: sl1 | l1 | l2')  # sl1: smooth L1 loss
        self.parser.add_argument('--hm_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint local offsets.')

        self.parser.add_argument('--wh_weight',
                                 type=float,
                                 default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss',
                                 default='ce',
                                 help='reid loss: ce | triplet')
        self.parser.add_argument('--id_weight',
                                 type=float,
                                 default=1,  # 0 for detection only and 1 for detection and re-id
                                 help='loss weight for id')  # ReID feature extraction or not
        

        self.parser.add_argument('--reid_dim',
                                 type=int,
                                 default=256,
                                 # KPI: track re-acquisition ≤ 3s after 5s occlusion + retention ≥ 80%
                                 # after appearance change (jacket removal, direction reversal).
                                 # 128-d embedding has insufficient cosine separation for these constraints.
                                 # 256-d provides measurably better Rank-1 accuracy under appearance change.
                                 help='ReID embedding dimension. '
                                      '256 recommended for KPI: track re-acquisition ≤ 3s + '
                                      'retention ≥ 80% after appearance change.')
        self.parser.add_argument('--input-wh',
                                 type=lambda s: tuple(int(x) for x in s.split(',')),
                                 default=(1280, 704),
                                 # KPI-derived resolution for SIYI ZR10 + ≤ 200ms onboard latency:
                                 #   Native: 2560×1440 (16:9),  pixel pitch ≈ 2.5 μm
                                 #   Person @ 500m standoff, 100m AGL, 10× zoom → 36px at 1280×704 ✓
                                 #   Vehicle @ 800m standoff, 10× zoom          → 54px at 1280×704 ✓
                                 #   Person @ 100m AGL, 1× zoom                 → 15px at 1280×704 ✓ min
                                 #   Inference (lwdetr_small, Jetson Orin NX): ~90–140ms ✓ fits 200ms budget
                                 #   Aspect ratio: 1280/704=1.818 vs ZR10 native 1.778 (2.2% diff)
                                 #   Both dims divisible by 64: 1280/64=20 ✓, 704/64=11 ✓
                                 #   -- lwdetr_tiny @ 1088×640: ~50–90ms if tighter latency needed
                                 help='net input resolution as W,H (both must be divisible by 64). '
                                      'KPI balanced: 1280,704 (≤200ms + 500m person detection). '
                                      'Fastest option: 1088,640 (~70ms, zoom-mode only).')

        # ----------------------1~10 object classes are what we need
        # pedestrian      (1),  --> 0
        # people          (2),  --> 1
        # bicycle         (3),  --> 2
        # car             (4),  --> 3
        # van             (5),  --> 4
        # truck           (6),  --> 5
        # tricycle        (7),  --> 6
        # awning-tricycle (8),  --> 7
        # bus             (9),  --> 8
        # motor           (10), --> 9
        # ----------------------

        # others          (11)
        self.parser.add_argument('--reid_cls_ids',
                                 default='0,1,2,3,4,5,6,7,8,9',  # '0,1,2,3,4' or '0,1,2,3,4,5,6,7,8,9'
                                 help='')  # the object classes need to do reid

        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh',
                                 action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset',
                                 action='store_true',
                                 help='not regress local offset.')

        self.parser.add_argument("--local-rank",
                                 type=int,
                                 default=-1,
                                 help="Local rank for distributed training. Set automatically by torchrun/distributed.launch.")

        self.parser.add_argument('--save_dir_result',
                                 default='Lun_6_test_track_try',
                                 type=str)

        self.parser.add_argument('--tri',
                                 action='store_true')

        self.parser.add_argument('--use_repeat_sampling',
                                 action='store_true',
                                 help='Use repeat factor sampling to over-sample rare classes.')
        self.parser.add_argument('--repeat_thresh',
                                 type=float,
                                 default=0.001,
                                 help='Frequency threshold t for repeat factor: rf=sqrt(t/f(c)).')



    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        # opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        # print("opt.gpus", opt.gpus)
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

        opt.reg_offset = not opt.not_reg_offset

        # Hybrid CenterNet head uses upsampled stride-4 feature → same down_ratio as MOT
        # (CenterNetUpsampleNeck in model.py handles the stride-16 → stride-4 upsampling)

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        """
        :param opt:
        :param dataset:
        :return:
        """
        input_h, input_w = dataset.default_input_wh  # 图片的高和宽
        opt.mean, opt.std = dataset.mean, dataset.std  # 均值 方差
        opt.num_classes = dataset.num_classes  # 类别数
        print('num_classes:', opt.num_classes)
        for reid_id in opt.reid_cls_ids.split(','):
            if int(reid_id) > opt.num_classes - 1:
                print('[Err]: configuration conflict of reid_cls_ids and num_classes!')
                return

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        # ViT patch embed requires dims divisible by 64 — round up silently
        opt.input_h = ((opt.input_h + 63) // 64) * 64
        opt.input_w = ((opt.input_w + 63) // 64) * 64
        if opt.input_h != (opt.input_h // 64 * 64) or opt.input_w != (opt.input_w // 64 * 64):
            print(f'[warn] input size snapped to {opt.input_w}×{opt.input_h} (must be divisible by 64)')
        opt.output_h = opt.input_h // opt.down_ratio  # 输出特征图的宽高
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'mot':
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes,
                         'id': opt.reid_dim}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            if opt.id_weight > 0:
                opt.nID_dict = dataset.nID_dict

        elif opt.task == 'hybrid':
            # HybridCenterNetDETR builds its own heads from HybridModelConfig;
            # heads dict is unused but populated to keep the pipeline uniform.
            opt.heads = {}
            if opt.id_weight > 0:
                opt.nID_dict = dataset.nID_dict

        else:
            raise ValueError(f'Unknown task: {opt.task!r}. Supported: mot | hybrid')

        print('heads: ', opt.heads)
        return opt

    def init(self, args=''):
        opt = self.parse(args)

        use_imagenet_norm = 'lwdetr' in opt.arch or 'hybrid' in opt.arch
        _common = {
            'default_input_wh': [opt.input_wh[1], opt.input_wh[0]],
            'num_classes': len(opt.reid_cls_ids.split(',')),
            'mean': [0.485, 0.456, 0.406] if use_imagenet_norm else [0.408, 0.447, 0.470],
            'std':  [0.229, 0.224, 0.225] if use_imagenet_norm else [0.289, 0.274, 0.278],
            'dataset': 'jde',
            'nID': 14455,
            'nID_dict': {},
        }
        default_dataset_info = {
            'mot':    _common,
            'hybrid': _common,
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        h_w = default_dataset_info[opt.task]['default_input_wh']
        opt.img_size = (h_w[1], h_w[0])
        print('Net input image size: {:d}×{:d}'.format(h_w[1], h_w[0]))

        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)

        return opt
