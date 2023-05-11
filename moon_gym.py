import os
from abc import *
from enum import Enum, auto
import numpy as np

import gym
from gym import spaces
from smac.env import StarCraft2Env

# Configure dm_control to use the EGL rendering backend (requires GPU)
import os
from PIL import Image

class MoonGymFramework(Enum):
    SMAC = auto()

class SMACObservationSpace:
    def __init__(self, obs_shape, state_shape, n_agents):
        self.n_agents = n_agents
        self.shape = (state_shape + obs_shape, )

    def get_obs(self, state, obs):
        mobs = []
        for idx in range(self.n_agents):
            mobs.append(np.concatenate((state, obs[idx])))

        return mobs

class SMACActionSpace:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.shape = (self.n_actions,)

    def sample(self):
        return random.choice(range(self.n_actions))

class MoonGym:
    def __init__(self, env="8m",
            seed=42, framework=MoonGymFramework.SMAC):

        print ("*" * 80)
        print ("Framework : ", framework.name)
        print ("env : ", env)
        print ("*" * 80)

        self.env = env
        self.framework = framework

        assert self.framework == MoonGymFramework.SMAC
        self.mgym = StarCraft2Env(map_name=env)

        env_info = self.mgym.get_env_info()

        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.obs_shape = env_info['obs_shape']
        self.state_shape = env_info['state_shape']

        self.action_space = SMACActionSpace(self.n_actions)
        print ("self.action_space.shape : ", self.action_space.shape)
        self.observation_space = SMACObservationSpace(env_info['obs_shape'], env_info['state_shape'], self.n_agents)

    def __repr__(self):
        return "SMAC(%s) : %d actions/%d agents/%d obs shape/%d state shape" %\
            (self.env, self.n_actions, self.n_agents, self.obs_shape, self.state_shape)

    def __del__(self):
        self.mgym.close()

    def _get_obs(self):
        obs = self.mgym.get_obs()
        state = self.mgym.get_state()
        return self.observation_space.get_obs(state, obs)

    def reset(self):
        assert self.framework == MoonGymFramework.SMAC
        self.mgym.reset()

        return self._get_obs()

    def choose_actions(self, actions):
        result = []
        for agent_idx in range(self.n_agents):
            mask_val = self.mgym.get_avail_agent_actions(agent_idx)
            mask_val = list(map(lambda x: 0 if x ==1 else 1, mask_val))
            masked_actions = np.ma.masked_array(actions[agent_idx], 
                mask=mask_val)
            result.append(masked_actions.argmax())

        return result

    def step(self, actions):
        assert self.framework == MoonGymFramework.SMAC
        reward, terminated, _ = self.mgym.step(self.choose_actions(actions))

        return self._get_obs(), reward, terminated

    def save_replay(self):
        self.mgym.save_replay()

if __name__ == "__main__":
    mgym = MoonGym(framework=MoonGymFramework.ORIGINAL_GYM)
    res = mgym.step(mgym.action_space.sample())
