from enum import Enum

LOG_LEVEL = 3                       # minimum log level to print to terminal. see LogLevel below

BATCH_SIZE = 128                    # batch size for training and prediction
EPOCHS = 40                         # epochs to train models and sub-models

#   exploration parameters
MAX_TRAINING_STEPS = 300            # max steps per episode
SESSION_COUNT = 5
EPISODE_COUNT = 200

#   epsilon exploration parameters
EPSILON_START = 1.0
EPSILON_DECAY = 0.975
EPSILON_MIN = 0.01

#   environment parameters
ACTION_SIZE = 2
OBSERVATION_SIZE = 4


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    SILENT = 3