
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from algo2.ddpg2.model import (Actor, Critic)
from algo2.ddpg2.memory import SequentialMemory
from algo2.ddpg2.random_process import OrnsteinUhlenbeckProcess
from algo2.ddpg2.util import *

from algo2.wrapper import RLAlgoWrapper
from functools import reduce
from util.env import args, device

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(RLAlgoWrapper):
    def __init__(self, env, agent_count, load_model_path):
        # (self, nb_states, nb_actions, args):
        super().__init__(env, agent_count, load_model_path)

        seed = args.seed + 10000

        self.hidden1 = 400
        self.hidden2 = 300
        self.init_w = 0.003
        self.prate = 0.0001
        self.rate = 0.001
        self.rmsize = 6000000
        self.window_length = 1
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_mu = 0.0
        self.bsize = 64
        self.tau = 0.001
        self.discount = 0.99
        self.epsilon = 50000
        self.warmup = 100
        self.step = 0

        if seed > 0:
            self.seed(seed)
        
        self.nb_states = env.observation_space.shape[0]
        self.nb_actions = env.action_space.shape[0]
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':self.hidden1, 
            'hidden2':self.hidden2, 
            'init_w':self.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=self.rmsize, window_length=self.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=self.ou_theta, mu=self.ou_mu, sigma=self.ou_sigma)

        # Hyper-parameters
        self.batch_size = self.bsize
        self.tau = self.tau
        self.discount = self.discount
        self.depsilon = 1.0 / self.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def _update(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float64))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action
    
    def act(self, state, is_train):
        if self.step <= self.warmup:
            action = self.random_action()
            self.step += 1
        else:
            action = self.select_action(state)

        return action
    
    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

    def handle_step(self, state, action, reward, next_state, is_done, timeout, epoch_ended):
         # agent observe and update policy
        self.observe(reward, state, is_done)
        # if self.step > self.warmup :
        #     self._update()

    def handle_epoch(self): 
        self._update()

    def get_model(self):
            return self.actor