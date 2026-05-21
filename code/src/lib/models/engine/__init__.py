"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

# Register model components only (skip data/optim/solver to avoid heavy deps)
from . import deim

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)