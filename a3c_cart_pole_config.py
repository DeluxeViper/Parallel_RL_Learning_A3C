import torch

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
#     SHARE_MEM_DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
#     SHARE_MEM_DEVICE = torch.device('cpu')  # share memory is not supported on mps
# else:
SHARE_MEM_DEVICE = torch.device('cpu')
DEVICE = torch.device('cpu')

MAX_MEMORY = 1000
NUM_EPISODES = 20000
max_episode = 30000
lr = 0.00001
log_interval = 10
n_step = 10
env_name = "CartPole-v1"
NUM_ACTORS = 2
LR = 1e-4
GAMMA = 0.9
BATCH_SIZE = 64
UPDATE_TARGET = 1000

EPSILON = 1.0
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.01