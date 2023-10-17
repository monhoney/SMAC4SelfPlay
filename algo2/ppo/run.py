import os
import numpy as np
from functools import reduce
from torch.optim import Adam

from algo2.wrapper import RLAlgoWrapper
from algo2.ppo.ppo_buffer import PPOBuffer
from util.env import args, device
import algo2.ppo.core as core
import torch

class PPO(RLAlgoWrapper):
    def __init__(self, env, agent_count, load_model_path):
        super().__init__(env, agent_count, load_model_path)

        # Initialize Hyper Parameters
        self.obs_dim = env.observation_space.shape
        self.obs_flat_dim = (reduce (lambda x,y:x*y, 
            env.observation_space.shape),)
        self.act_dim = env.action_space.shape
        self.n_agents = env.n_agents
        self.local_steps_per_epoch = args.steps

        self.gamma=0.99 
        self.clip_ratio=0.2
        self.pi_lr=3e-4
        self.vf_lr=1e-3
        self.train_pi_iters=80 
        self.train_v_iters=80
        self.lam=0.97
        self.target_kl=0.01

        # PPO 버퍼 설정
        self.buf = PPOBuffer(self.obs_flat_dim, self.act_dim, 
            self.local_steps_per_epoch, self.gamma, self.lam)

        # Actor-Critic 생성
        self.ac = core.MLPActorCritic(env.observation_space, env.action_space)

        if load_model_path != "":
            print ("model is loaded!!! (path=%s)" % load_model_path)
            self.ac.load_state_dict(torch.load(load_model_path))

        # Optimizer 설정
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def get_model(self):
        return self.ac

    def act(self, obs, is_train):
        a, v, logp = self.ac.step(torch.as_tensor(obs,
            dtype=torch.float32).to(device))

        self.last_a = a
        self.last_v = v
        self.last_logp = logp

        return a

    def handle_step(self, obs, a, reward, next_obs, is_done, timeout, epoch_ended):
        self.buf.store(obs, a, reward, self.last_v, self.last_logp)

        # 트레젝토리 끝단 처리
        if is_done or timeout or epoch_ended:
            if timeout or epoch_ended:
                _, v, _ = self.ac.step(torch.as_tensor(next_obs,
                    dtype=torch.float32).to(device))
            else:
                v = 0

            self.buf.finish_path(v)

    def handle_epoch(self): 
        self._update()

    # Set up function for computing PPO policy loss
    def _compute_loss_pi(self, data):
        obs, act, adv = data['obs'], data['act'], data['adv']
        logp_old = data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, 
            dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def _compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()

    def _update(self):
        pi_l_old_list = []
        v_l_old_list = []
        kl_list = []
        ent_list = []
        cf_list = []
        delta_loss_pi_list = []
        delta_loss_v_list = []

        data = self.buf.get()

        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break

            loss_pi.backward()
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)

            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        cf = pi_info['cf']

        pi_l_old_list.append(pi_l_old)
        v_l_old_list.append(v_l_old)
        kl_list.append(kl)
        ent_list.append(ent)
        cf_list.append(cf)
        delta_loss_pi_list.append(loss_pi.item() - pi_l_old)
        delta_loss_v_list.append(loss_v.item() - v_l_old)

#       self.logger.store(LossPi=np.mean(pi_l_old_list), 
#           LossV=np.mean(v_l_old_list),
#           KL=np.mean(kl_list),
#           Entropy=np.mean(ent_list),
#           ClipFrac=np.mean(cf_list),
#           DeltaLossPi=np.mean(delta_loss_pi_list),
#           DeltaLossV=np.mean(delta_loss_v_list))
