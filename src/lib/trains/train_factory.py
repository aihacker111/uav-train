from .mot import MotTrainer
from .hybrid import HybridTrainer

train_factory = {
    'mot':    MotTrainer,
    'hybrid': HybridTrainer,
}
