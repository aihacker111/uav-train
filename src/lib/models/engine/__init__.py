"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

# for register purpose
from . import optim
from . import data
from . import deim

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)

# MOT extensions — triggers @register() for DEIMMotCriterion, VisDroneDataset, MotSolver
from .deim.deim_mot_criterion import DEIMMotCriterion   # noqa: F401
from .solver.mot_solver import MotSolver                # noqa: F401