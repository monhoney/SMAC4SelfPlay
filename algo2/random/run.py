import os
import numpy as np
from algo2.wrapper import RLAlgoWrapper

class RANDOM(RLAlgoWrapper):
    def __init__(self, env, agent_count, load_model_path):
        super().__init__(env, agent_count, load_model_path)

        # Initialize Hyper Parameters
        self.act_dim = env.action_space.shape

    def get_model(self):
        # should not be called!!!
        assert False

    def act(self, obs, is_train):
        return np.array(np.random.rand(self.act_dim[0]), dtype=np.float32)

    def handle_step(self, obs, a, reward, next_obs, is_done, timeout, epoch_ended):
        pass

    def handle_epoch(self): 
        pass
