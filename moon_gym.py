import os
from abc import *
from enum import Enum, auto
import numpy as np
import random

import gym
from gym import spaces
from smac.env import StarCraft2Env, StarCraft2SPEnv

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
        #return list(map(lambda x: random.random(), range(self.n_actions)))
        #return list(map(lambda x: random.choice(range(10)), range(4)))

class MoonGym:
    def __init__(self, env="8m",
            seed=42, framework=MoonGymFramework.SMAC,
            selfplay=False):

        print ("*" * 80)
        print ("Framework : ", framework.name)
        print ("env : ", env)
        print ("*" * 80)

        self.env = env
        self.framework = framework
        self.selfplay = selfplay

        assert self.framework == MoonGymFramework.SMAC
        
        if selfplay == True:
            self.mgym = StarCraft2SPEnv(map_name=env)
        else:
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

        if self.selfplay == True:
            return [self.observation_space.get_obs(state[0], obs[0]),
                self.observation_space.get_obs(state[0], obs[0])]
        else:
            return self.observation_space.get_obs(state, obs)

    def reset(self):
        assert self.framework == MoonGymFramework.SMAC
        self.mgym.reset()

        return self._get_obs()

    def choose_actions(self, actions, player_id=None):

        result = []
        if self.selfplay == True:
            for agent_idx in range(self.n_agents):
                mask_val = self.mgym.get_avail_agent_actions(player_id, agent_idx)
                mask_val = list(map(lambda x: 0 if x ==1 else 1, mask_val))
                masked_actions = np.ma.masked_array(actions[agent_idx], 
                    mask=mask_val)
                result.append(masked_actions.argmax())
        else:
            for agent_idx in range(self.n_agents):
                mask_val = self.mgym.get_avail_agent_actions(agent_idx)
                mask_val = list(map(lambda x: 0 if x ==1 else 1, mask_val))
                masked_actions = np.ma.masked_array(actions[agent_idx], 
                    mask=mask_val)
                result.append(masked_actions.argmax())

        return result

    def step(self, actions):
        assert self.framework == MoonGymFramework.SMAC

        if self.selfplay == True:
            reward, terminated, info = self.mgym.step(\
                [self.choose_actions(actions[0], player_id=0),
                self.choose_actions(actions[1], player_id=1)])
            return self._get_obs(), reward, terminated, info
        else:
            reward, terminated, info = self.mgym.step(self.choose_actions(actions))
            return self._get_obs(), reward, terminated, info

    def save_replay(self):
        self.mgym.save_replay()

if __name__ == "__main__":
    mgym = MoonGym(env='Simple64_Tank', framework=MoonGymFramework.SMAC, selfplay=True)

    mgym.reset()
    for _ in range(10000):
        mgym.step([list(map(lambda x: random.choice(range(10)), range(4))), 
        list(map(lambda x: random.choice(range(10)), range(4)))])
