from .mot import MotTrainer
from .hybrid import HybridTrainer
from .deim_mot import DeimMotTrainer
from .detr_mot import DetrMotTrainer

train_factory = {
    'mot':        MotTrainer,
    'hybrid':     HybridTrainer,
    'deim_mot':   DeimMotTrainer,
    'deimv2_jde': DetrMotTrainer,
}
