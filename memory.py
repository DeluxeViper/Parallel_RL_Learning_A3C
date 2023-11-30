import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.position = 0

    def push(self, state, next_state, action, reward, mask):
        # """Saves a transition."""
        self.buffer.append(Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        # transitions = self.buffer
        sampled_transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*sampled_transitions))
        return batch

    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        self.buffer.clear()
