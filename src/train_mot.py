"""
DEIMv2 MOT training entry point.

Identical to DEIMv2/train.py but:
  - adds src/lib/models to sys.path so `import engine` resolves our extended engine
  - imports engine (triggers @register() for DEIMMotCriterion, VisDroneDataset, MotSolver)
  - then delegates to DEIMv2's DetSolver / MotSolver via TASKS[cfg.task]

Usage (single GPU):
  python src/train_mot.py -c src/configs/deimv2_dinov3_s_visdrone.yml \\
                          -t models/deimv2_dinov3_s_coco.pth

Usage (multi-GPU, 4 GPUs):
  torchrun --nproc_per_node=4 src/train_mot.py \\
      -c src/configs/deimv2_dinov3_s_visdrone.yml \\
      -t models/deimv2_dinov3_s_coco.pth

  -t / --tuning  : load DEIMv2 COCO pretrained weights (fine-tune)
  -r / --resume  : resume from a checkpoint
  --test-only    : run evaluation only (no training)
"""
import os
import sys

# Make 'engine' importable from src/lib/models/engine
_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'models')
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import argparse

import engine  # triggers all @register() decorators (backbone, deim, data, optim, ...)
               # also imports DEIMMotCriterion, VisDroneDataset, MotSolver via engine/__init__.py

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS   # {'detection': DetSolver, 'mot': MotSolver, ...}


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only one of --tuning / --resume at a time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items()
                        if k not in ['update'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    # Disable HGNetv2 pretrained download when fine-tuning / resuming
    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg:', cfg.__dict__)

    task = cfg.yaml_cfg.get('task', 'detection')
    if task not in TASKS:
        raise ValueError(f'Unknown task {task!r}. Available: {list(TASKS.keys())}')

    solver = TASKS[task](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2 MOT training')

    # priority 0
    parser.add_argument('-c', '--config',    type=str, default='', help='YAML config path')
    parser.add_argument('-r', '--resume',    type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning',    type=str, help='fine-tune from checkpoint (COCO → VisDrone)')
    parser.add_argument('-d', '--device',    type=str, help='device (e.g. cuda:0)')
    parser.add_argument('--seed',            type=int, default=0)
    parser.add_argument('--use-amp',         action='store_true')
    parser.add_argument('--output-dir',      type=str)
    parser.add_argument('--summary-dir',     type=str)
    parser.add_argument('--test-only',       action='store_true', default=False)

    # priority 1 — override any YAML key from CLI
    parser.add_argument('-u', '--update', nargs='+', help='override YAML keys, e.g. epoches=20')

    # distributed
    parser.add_argument('--print-method', type=str, default='builtin')
    parser.add_argument('--print-rank',   type=int, default=0)
    parser.add_argument('--local-rank',   type=int, help='set by torchrun')

    args = parser.parse_args()
    main(args)
