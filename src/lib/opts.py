from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--task', default='hybrid', help='hybrid')
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
                                 default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--device',
                                 default='cuda:0',
                                 help='device string passed to torch (e.g. cuda:0, cpu)')
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=8,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')
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

        # model
        self.parser.add_argument('--arch',
                                 default='hybrid_ecdet',
                                 help='model architecture. "hybrid" activates HybridECDet path.')
        self.parser.add_argument('--ecdet_config',
                                 type=str,
                                 default='',
                                 help='Path to ECDet YAML config. Required for hybrid architectures.')
        self.parser.add_argument('--head_conv',
                                 type=int,
                                 default=32,
                                 help='intermediate conv channels for CenterNet auxiliary head. '
                                      '0 disables the intermediate conv (1×1 only).')
        self.parser.add_argument('--backbone_lr_scale',
                                 type=float,
                                 default=0.05,
                                 help='LR multiplier for ViT backbone (relative to head LR).')
        self.parser.add_argument('--freeze_backbone_epochs',
                                 type=int,
                                 default=0,
                                 help='Number of epochs to freeze ViT backbone before fine-tuning. 0 = no freeze.')
        self.parser.add_argument('--backbone_weights',
                                 type=str,
                                 default='',
                                 help='Path to pretrained ECDet checkpoint (.pth). Empty = train from scratch.')
        self.parser.add_argument('--grad_checkpoint',
                                 action='store_true',
                                 default=False,
                                 help='Enable gradient checkpointing on the ViT backbone. '
                                      'Reduces backbone VRAM by ~50%% at ~20%% slower backward.')
        self.parser.add_argument('--down_ratio',
                                 type=int,
                                 default=4,
                                 help='Output stride used by the dataset to compute heatmap size.')

        # input
        self.parser.add_argument('--input_res',
                                 type=int,
                                 default=-1,
                                 help='input height and width. -1 for default from dataset.')
        self.parser.add_argument('--input_h',
                                 type=int,
                                 default=-1,
                                 help='input height override. -1 = use dataset default.')
        self.parser.add_argument('--input_w',
                                 type=int,
                                 default=-1,
                                 help='input width override. -1 = use dataset default.')

        # train
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=5e-4,
                                 help='learning rate for heads/decoder. '
                                      'Backbone gets backbone_lr_scale × lr.')
        self.parser.add_argument('--lr_scale',
                                 type=str,
                                 default='linear',
                                 choices=['linear', 'sqrt', 'none'],
                                 help='Multi-GPU LR scaling rule.')
        self.parser.add_argument('--base_batch_size',
                                 type=int,
                                 default=4,
                                 help='Reference batch size the base --lr was tuned for.')
        self.parser.add_argument('--lr_step',
                                 type=str,
                                 default='20, 27',
                                 help='Epochs to drop LR by 10× (step decay, used when cosine_lr=False).')
        self.parser.add_argument('--cosine_lr',
                                 default=True,
                                 action=argparse.BooleanOptionalAction,
                                 help='Use cosine LR with linear warmup. Use --no-cosine-lr to disable.')
        self.parser.add_argument('--warmup_iters',
                                 type=int,
                                 default=2000,
                                 help='Linear LR warmup optimizer-steps (cosine_lr only).')
        self.parser.add_argument('--min_lr_ratio',
                                 type=float,
                                 default=0.01,
                                 help='Minimum LR as fraction of base LR at end of cosine decay.')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=30,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=4,
                                 help='batch size per GPU.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='iterations per epoch. -1 = len(dataset) / batch_size.')
        self.parser.add_argument('--use_imagenet_norm',
                                 action='store_true',
                                 default=True,
                                 help='Apply ImageNet mean/std normalization. Always True for ECViT.')
        self.parser.add_argument('--use_amp',
                                 action='store_true',
                                 default=False,
                                 help='Enable AMP (fp16 forward + fp32 weights).')
        self.parser.add_argument('--grad_clip',
                                 type=float,
                                 default=0.1,
                                 help='Max gradient norm. 0.0 to disable.')
        self.parser.add_argument('--weight_decay',
                                 type=float,
                                 default=1e-4,
                                 help='AdamW weight decay for encoder and head param groups.')
        self.parser.add_argument('--backbone_wd',
                                 type=float,
                                 default=0.01,
                                 help='AdamW weight decay for backbone param group.')
        self.parser.add_argument('--encoder_lr_scale',
                                 type=float,
                                 default=0.5,
                                 help='LR multiplier for encoder param group relative to head LR.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs between validation runs.')
        self.parser.add_argument('--trainval',
                                 action='store_true',
                                 help='include validation in training and test on test set')
        self.parser.add_argument('--grad_accum', type=int, default=1,
                                 help='Gradient accumulation steps. Effective batch = batch_size * grad_accum.')

        # evaluation
        self.parser.add_argument('--score_thr',
                                 type=float,
                                 default=0.3,
                                 help='Detection score threshold for val evaluation.')
        self.parser.add_argument('--K',
                                 type=int,
                                 default=200,
                                 help='max number of output objects per image.')

        # tracking (used by track_AMOT.py)
        self.parser.add_argument('--test_visdrone', action='store_true', default=True,
                                 help='run tracking eval on VisDrone test-dev sequences')
        self.parser.add_argument('--test_uavdt', action='store_true',
                                 help='run tracking eval on UAVDT sequences')
        self.parser.add_argument('--min_box_area', type=float, default=100,
                                 help='filter out tiny boxes below this pixel area')
        self.parser.add_argument('--conf_thres', type=float, default=0.4,
                                 help='confidence threshold for tracking')
        self.parser.add_argument('--det_thres', type=float, default=0.3,
                                 help='confidence threshold for detection')
        self.parser.add_argument('--nms_thres', type=float, default=0.4,
                                 help='IoU threshold for NMS')
        self.parser.add_argument('--track_buffer', type=int, default=30,
                                 help='tracking buffer length in frames')
        self.parser.add_argument('--save_dir_result', default='results', type=str,
                                 help='directory for tracking output')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep original resolution during tracking inference')

        # dataset config
        self.parser.add_argument('--data_cfg', type=str,
                                 default='../src/lib/cfg/visdrone.json',
                                 help='path to dataset JSON config')
        self.parser.add_argument('--data_dir', type=str,
                                 default='/media/jianbo/ioe/UAVdata',
                                 help='root directory of the dataset')

        # ── HybridLoss weights ────────────────────────────────────────────────────
        self.parser.add_argument('--bbox_weight', type=float, default=5.0,
                                 help='SmoothL1 bounding-box loss weight (lambda_bbox).')
        self.parser.add_argument('--giou_weight', type=float, default=2.0,
                                 help='CIoU loss weight (lambda_ciou).')
        self.parser.add_argument('--dn_weight', type=float, default=1.0,
                                 help='Denoising loss weight (lambda_dn).')
        self.parser.add_argument('--dn_warmup_epochs', type=int, default=10,
                                 help='Epochs over which DN loss weight ramps from 0.5×λ_dn → λ_dn.')
        self.parser.add_argument('--enc_aux_weight', type=float, default=0.25,
                                 help='Encoder top-K supervision loss weight.')
        self.parser.add_argument('--mal_gamma', type=float, default=2.0,
                                 help='Exponent for MAL (Modulation Augmented Loss) quality target.')
        # ── Heatmap (CenterNet) branch losses ────────────────────────────────
        self.parser.add_argument('--heatmap_weight', type=float, default=1.0,
                                 help='Weight for CenterNet focal heatmap loss (lambda_heatmap).')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='Weight for width/height L1 loss at GT cells (lambda_wh).')
        self.parser.add_argument('--reg_weight', type=float, default=1.0,
                                 help='Weight for sub-pixel offset L1 loss at GT cells (lambda_reg).')
        # ── Class-imbalance methods ───────────────────────────────────────────
        self.parser.add_argument('--logit_adj_tau', type=float, default=0.5,
                                 help='Logit Adjustment (ICLR 2021) temperature tau. '
                                      'Adds tau*log(pi_c) to logits at train time to correct for '
                                      'class-frequency prior. 0 = disabled.')
        self.parser.add_argument('--aux_loss',
                                 default=True,
                                 action=argparse.BooleanOptionalAction,
                                 help='Auxiliary losses from intermediate decoder layers. '
                                      'Use --no-aux_loss to disable.')

        # ── ReID ──────────────────────────────────────────────────────────────────
        self.parser.add_argument('--id_weight', type=float, default=1.0,
                                 help='ReID loss weight. 0 = detection only.')
        self.parser.add_argument('--triplet_weight', type=float, default=0.5,
                                 help='Triplet loss weight within ReID. L_reid = L_ce + triplet_weight * L_tri.')
        self.parser.add_argument('--reid_dim',
                                 type=int,
                                 default=256,
                                 help='ReID embedding dimension.')
        self.parser.add_argument('--reid_cls_ids',
                                 default='0,1,2,3,4,5,6,7,8,9',
                                 help='object class IDs to perform ReID on')

        self.parser.add_argument('--input-wh',
                                 type=lambda s: tuple(int(x) for x in s.split(',')),
                                 default=(1280, 704),
                                 help='net input resolution as W,H (both must be divisible by 64). '
                                      'Default 1280,704 for SIYI ZR10.')

        self.parser.add_argument("--local-rank",
                                 type=int,
                                 default=-1,
                                 help="Local rank for distributed training (set by torchrun).")

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
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        # ECViT backbone always requires ImageNet normalization.
        if 'hybrid' in opt.arch:
            opt.use_imagenet_norm = True

        import torch
        if not torch.cuda.is_available() and 'cuda' in opt.device:
            print(f'[opts] CUDA not available, falling back device {opt.device!r} → cpu')
            opt.device = 'cpu'

        if opt.trainval:
            opt.val_intervals = 100000000

        n = len(opt.gpus)
        q, r = divmod(opt.batch_size, max(n, 1))
        opt.chunk_sizes = [q + (1 if i < r else 0) for i in range(max(n, 1))]
        print('chunk_sizes:', opt.chunk_sizes)

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
        input_h, input_w = dataset.default_input_wh
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        print('num_classes:', opt.num_classes)
        for reid_id in opt.reid_cls_ids.split(','):
            if int(reid_id) > opt.num_classes - 1:
                print('[Err]: configuration conflict of reid_cls_ids and num_classes!')
                return

        # input_h/w: CLI arg overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        # ViT patch embed requires dims divisible by 64 — round up silently
        opt.input_h = ((opt.input_h + 63) // 64) * 64
        opt.input_w = ((opt.input_w + 63) // 64) * 64

        if opt.task == 'hybrid':
            opt.heads = {}
            if opt.id_weight > 0:
                opt.nID_dict = dataset.nID_dict
        else:
            raise ValueError(f'Unknown task: {opt.task!r}. Only "hybrid" is supported.')

        print('heads: ', opt.heads)
        return opt

    def init(self, args=''):
        opt = self.parse(args)

        use_imagenet_norm = 'hybrid' in opt.arch
        _common = {
            'default_input_wh': [opt.input_wh[1], opt.input_wh[0]],
            'num_classes': len(opt.reid_cls_ids.split(',')),
            'mean': [0.485, 0.456, 0.406] if use_imagenet_norm else [0.408, 0.447, 0.470],
            'std':  [0.229, 0.224, 0.225] if use_imagenet_norm else [0.289, 0.274, 0.278],
            'dataset': 'jde',
            'nID': 14455,
            'nID_dict': {},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        h_w = _common['default_input_wh']
        opt.img_size = (h_w[1], h_w[0])
        print('Net input image size: {:d}×{:d}'.format(h_w[1], h_w[0]))

        dataset = Struct(_common)
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)

        return opt
