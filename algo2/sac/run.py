import os
import numpy as np
from functools import reduce

from algo2.wrapper import RLAlgoWrapper
from util.env import args, device
import torch

from algo2.sac.networks import ActorCritic
from algo2.sac.buffer import ReplayMemory
'''
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
'''

class SAC(RLAlgoWrapper):
    def __init__(self, env, agent_count, load_model_path):
        super().__init__(env, agent_count, load_model_path)

        # self.gamma = 0.99
        # self.tau = 0.005
        # self.alpha = 0.2

        self.policy_type = "Gaussian" # or "Deterministic"
        # self.target_update_interval = 1
        # self.automatic_entropy_tuning = False

        self.device = device
        # self.lr = 0.0003
        self.hidden_size=256
        self.batch_size=256

        self.obs_dim = env.observation_space.shape
        self.obs_flat_dim = (reduce (lambda x,y:x*y, 
            env.observation_space.shape),)
        self.act_dim = env.action_space.shape
        self.act_space = env.action_space
        self.n_agents = env.n_agents
        self.local_steps_per_epoch = args.steps
        self.env = env
        
        self.replay_size = 10000000

        self.ac = ActorCritic(self.obs_dim[0], self.act_dim, self.hidden_size, 
                              device, self.policy_type, self.act_space)
        self.memory = ReplayMemory(self.replay_size, args.seed)
        self.policy = self.ac.policy
        
        if load_model_path != "":
            print ("model is loaded!!! (path=%s)" % load_model_path)
            self.ac.load_state_dict(torch.load(load_model_path))

    def get_model(self):
        return self.ac
    
    def act(self, obs, is_train):
        state = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if is_train:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def handle_step(self, state, action, reward, next_state, is_done, timeout, epoch_ended):
        mask = 1 if timeout else float(not is_done)
        self.memory.push(state, action, reward, next_state, mask)
        if len(self.memory) > self.batch_size:
            self._update(updates=False)

    def _update(self, updates=False):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = \
            self.ac.step(state_batch, action_batch, next_state_batch, mask_batch, reward_batch, updates)
    
    def handle_epoch(self): 
        self._update(updates=True)