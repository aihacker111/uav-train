from .hawkdet import HawkDetTrainer
from .mot import MotTrainer

train_factory = {
    'hawkdet':   HawkDetTrainer,
    'ecdet_jde': MotTrainer,
}
