import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a3c_cart_pole_config import GAMMA

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.v1 = nn.Linear(num_inputs, 128)
        self.fc_critic = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        # value = self.fc_critic(x)
        # return policy, value
        # return policy, self.fc_critic(x)
        v1 = torch.tanh(self.v1(input))
        values = self.fc_critic(v1)
        return policy, values

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        print('model action: {action}')
        return action
    
class GlobalModel(Model):
    def __init__(self, num_inputs, num_outputs):
        super(GlobalModel, self).__init__(num_inputs, num_outputs)

class LocalModel(Model):
    def __init__(self, num_inputs, num_outputs):
        super(LocalModel, self).__init__(num_inputs, num_outputs)

    def push_to_global_model(self, batch, global_model, global_optimizer):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.stack(batch.action)
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        policy, value = self.forward(states)
        policy = policy.view(-1, self.num_outputs)
        value = value.view(-1)

        _, last_value = self.forward(next_states[-1])

        running_return = last_value[0].data
        running_returns = torch.zeros(rewards.size())
        for t in reversed(range(0, len(rewards))):
            running_return = rewards[t] + GAMMA * running_return * masks[t]
            running_returns[t] = running_return


        td_error = running_returns - value.detach()
        log_policy = (torch.log(policy + 1e-10) * actions).sum(dim=1, keepdim=True)
        loss_policy = - log_policy * td_error
        loss_value = torch.pow(td_error, 2)
        entropy = (torch.log(policy + 1e-10) * policy).sum(dim=1, keepdim=True)

        loss = (loss_policy + loss_value - 0.01 * entropy).mean()

        # print(f'Pushing local to Global model loss: {loss}')
        global_optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(self.parameters(), global_model.parameters()):
            gp._grad = lp.grad
        global_optimizer.step()

        return loss

    def pull_from_global_model(self, global_model):
        self.load_state_dict(global_model.state_dict())
