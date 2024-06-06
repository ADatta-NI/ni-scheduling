from os.path import expanduser
from typing import Final
from enum import Enum
from collections import OrderedDict

HYPHEN: Final[str] = '-'
NOOP: Final[str] = 'no-op'
NA: Final[str] = 'na'
MS_AGENT: Final[str] = 'MSAgent'
OS_AGENT_PREFIX: Final[str] = 'OSAgent-'
MS_POLICY: Final[str] = 'ms-policy'
OS_POLICY: Final[str] = 'os-policy'
CHECKPOINT_ROOT: Final[str] = expanduser("~") + '/checkpoints/'
RAY_RESULTS_ROOT: Final[str] = expanduser("~") + '/ray_results/'
PPO_CHECKPOINT_ROOT: Final[str] = expanduser("~") + '/saved_runs/checkpoints/'
SCHEDULING_ENV_RANDOM_SEED: Final[int] = 20231125
DATA_GENERATOR_TRAIN_SEED: Final[int] = 20230910
DATA_GENERATOR_TEST_SEED: Final[int] = 30331920

class JobStatus(Enum):
    NOT_STARTED     = 1
    IN_PROGRESS     = 2
    COMPLETED       = 3
    FAILED          = 4


class OperationStatus(Enum):
    NOT_STARTED     = 1
    IN_GLOBAL_BAG   = 2
    IN_LOCAL_BAG    = 3
    IN_PROGRESS     = 4
    COMPLETED       = 5
    FAILED          = 6
    
class TesterStatus(Enum):
    IDLE            = 1
    BUSY            = 2
    MAINTENANCE     = 3

class TimeJumpReason(Enum):
    NO_JUMP         = 1
    NEW_ARRIVAL     = 2
    TEST_DONE       = 3
