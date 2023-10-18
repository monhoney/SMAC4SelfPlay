import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from algo2.wrapper import RLAlgoWrapper
from functools import reduce
from util.env import args, device
from algo2.ddpg.ounoise import OUNoise
from algo2.ddpg.replay_memory import ReplayMemory, Transition
import numpy as np

def MSELoss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)


    def forward(self, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        mu = F.tanh(self.mu(x))
        return mu

    
class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear_action = nn.Linear(num_outputs, hidden_size)
        self.bn_a = nn.BatchNorm1d(hidden_size)
        self.bn_a.weight.data.fill_(1)
        self.bn_a.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        a = F.tanh(self.linear_action(actions))
        x = torch.cat((x, a), 1)
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        return V


class DDPG(RLAlgoWrapper):
    def __init__(self, env, agent_count, load_model_path):
        # (self, gamma, tau, hidden_size, num_inputs, action_space):
        super().__init__(env, agent_count, load_model_path)

        self.obs_dim = env.observation_space.shape
        self.obs_flat_dim = (reduce (lambda x,y:x*y, env.observation_space.shape),)
        self.act_dim = env.action_space.shape
        self.n_agents = env.n_agents
        self.local_steps_per_epoch = args.steps
        
        self.num_inputs = env.observation_space.shape[0]
        self.action_space = env.action_space
        hidden_size = 128

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.001

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        self.memory = ReplayMemory(1000000)
        self.ounoise = OUNoise(self.act_dim[0])
        self.ounoise.reset()
        self.epoch_change = 0
        self.batch_size = 128
        self.updates_per_step = 5


    def act(self, state, is_train, epoch):
        if self.epoch_change != epoch:
            self.ounoise.reset()
            self.epoch_change += 1
            print(self.epoch_change)
        exploration = None
        if epoch < args.epochs // 2:
            exploration = self.ounoise

        state_T = torch.Tensor(np.array([state]))
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor((Variable(state_T)))
        self.actor.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)

    def handle_step(self, state, action, reward, next_state, is_done, timeout, epoch_ended, epoch):
        if epoch < args.epochs // 2:
            state_T = torch.Tensor(np.array([state]))
            action_T = torch.Tensor(action)
            mask_T = torch.Tensor([not is_done])
            next_state_T = torch.Tensor(np.array([next_state]))
            reward_T = torch.Tensor([reward])
            
            self.memory.push(state_T, action_T, mask_T, next_state_T, reward_T)

            if len(self.memory) > self.batch_size * 5:
                for _ in range(self.updates_per_step):
                    transitions = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))

                    self._update(batch)
            if is_done:
                return
        else:
            pass
    
    def handle_epoch(self): 
        self._update()

    def _update(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        with torch.no_grad():
            next_state_batch = Variable(torch.cat(batch.next_state))
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = torch.unsqueeze(reward_batch, 1)
        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def get_model(self):
            return self.actor