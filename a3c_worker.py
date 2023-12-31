import gym
import torch
import torch.multiprocessing as mp
import numpy as np
from a3c_model import LocalModel
from memory import Memory
# from config import env_name, n_step, max_episode, log_interval
from a3c_cart_pole_config import env_name, n_step, max_episode, log_interval, MAX_MEMORY, BATCH_SIZE
import wandb
from collections import deque
import random

random_seed = 42 

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=20)
np.set_printoptions(precision=20)

# class ReplayMemory:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         # print(f'batch size: {batch_size}, buffer len {len(self.buffer)}')
#         s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
#         return np.stack(s), np.stack(a), np.stack(r), np.stack(ns), np.stack(d)
    
#     def clear(self):
#         self.buffer.clear()

#     def __len__(self):
#         return len(self.buffer)
    
# Equivalent to a worker
class DQNActor(mp.Process):
    def __init__(self, global_model, global_optimizer, global_ep, global_ep_r, res_queue, name, stop_event):
        super(DQNActor, self).__init__()

        self.memory = Memory(MAX_MEMORY)
        self.env = gym.make(env_name)
        self.name = 'w%i' % name
        self.global_ep, self.global_ep_r = global_ep, global_ep_r
        self.global_model, self.global_optimizer = global_model, global_optimizer
        self.local_model = LocalModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.num_actions = self.env.action_space.n
        self.stop_event = stop_event

    def get_action(self, policy, num_actions):
        # Converts an array that looks like [[1 1]] to [1, 1]
        # policy = policy.data.numpy()[0].flatten()
        # policy = policy.tolist() 
        policy = policy.data.numpy()[0]
        probabilities_sum = np.sum(policy)
        normalized_probabilities = [p / probabilities_sum for p in policy]
        action = np.random.choice(num_actions, 1, p=policy)[0]
        # print(f'action choice: {action}, normalized_probs: {normalized_probabilities}')
        return action

    def run(self):
        wandb.init(project=f'A3C_Baseline_{self.name}')
        total_steps = 0
        total_reward = 0

        while self.global_ep.value < max_episode:
            self.local_model.pull_from_global_model(self.global_model)
            done = False
            score = 0
            steps = 0

            state, info = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            # memory = Memory(n_step)
            truncated = False

            while not (done or truncated):
                # print(f'state: {state}, {state.squeeze(0)}')
                # policy, value = self.local_model(state.squeeze(0))
                policy, value = self.local_model(state.detach())
                action = self.get_action(policy, self.num_actions)

                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = torch.FloatTensor(next_state)
                next_state = next_state.unsqueeze(0)

                mask = 0 if done else 1
                reward = reward if not done or score == 499 else -1
                action_one_hot = torch.zeros(2)
                action_one_hot[action] = 1
                # print(f'memory: state {state}, next_state {next_state}, action_one_hot {action_one_hot} reward {reward} mask {mask}')
                self.memory.push(state, next_state, action_one_hot, reward, mask)

                score += reward
                state = next_state

                if len(self.memory) > BATCH_SIZE:
                    batch = self.memory.sample(BATCH_SIZE)
                    loss = self.local_model.push_to_global_model(batch, self.global_model, self.global_optimizer)

                    self.local_model.pull_from_global_model(self.global_model)
                    # memory = Memory(n_step)
                    self.memory.clear()

                if done:
                    # running_score = self.record(score, loss)
                    break
            if(truncated):
                truncate_counter +=1
            else:
                truncate_counter = 0
            self.global_ep.value +=1 # no lock is needed
            total_reward += score
            wandb.log({"Average reward": total_reward/(self.global_ep.value+1)})
            wandb.log({"Reward": score})
            print(f'Name: {self.name}, Global episode {self.global_ep.value}: Total reward {score}')

            if truncate_counter == 2:
                print("Target reward achieved, stopping training")
                self.stop_event.set()
                break

            if self.stop_event.is_set():
                break
        wandb.finish()