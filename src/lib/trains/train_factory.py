from .mot import MotTrainer
from .hybrid import HybridTrainer
from .deim_mot import DeimMotTrainer

train_factory = {
    'mot':      MotTrainer,
    'hybrid':   HybridTrainer,
    'deim_mot': DeimMotTrainer,
}
