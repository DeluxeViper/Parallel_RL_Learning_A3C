import gym
import glob
import torch
import torch.nn as nn
import numpy as np

from a3c_model import Model
from a3c_worker import DQNActor
from shared_adam import SharedAdam
from tensorboardX import SummaryWriter
from torch.multiprocessing import Process, Manager, Event
import torch.multiprocessing as mp

from collections import deque
import random
from a3c_cart_pole_config import *
from shared_adam import SharedAdam
import time
import wandb

# from config import env_name, lr

random_seed = 500

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    global_model = Model(num_inputs, num_actions)
    global_model.share_memory()
    global_optimizer = SharedAdam(global_model.parameters(), lr=LR)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    stop_event = Event()

    # writer = SummaryWriter('logs')

    # Note: online_model seems to = global_model and target model seems to
    #       = LocalModel
    workers = [DQNActor(global_model=global_model,
                        global_optimizer=global_optimizer,
                        global_ep=global_ep,
                        global_ep_r=global_ep_r,
                        res_queue=res_queue,
                        name=i,
                        stop_event=stop_event) for i in range(NUM_ACTORS)]
    # workers = [DQNActor(env_name, state_dim, action_dim, global_ep, global_model, global_optimizer, i, stop_event) for i in range(1)]
    # workers = [Worker(global_model, global_optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())] 
    [w.start() for w in workers]
    res = []
    [w.join() for w in workers]