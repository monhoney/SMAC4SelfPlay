import os
import numpy as np
from algo2.wrapper import RLAlgoWrapper

class COMPUTER(RLAlgoWrapper):
    def __init__(self, env, agent_count, load_model_path):
        super().__init__(env, agent_count, load_model_path)

        # Initialize Hyper Parameters
        self.act_dim = env.action_space.shape

    def get_model(self):
        # should not be called!!!
        assert False

    def act(self, obs, is_train):
        selected_option = np.zeros(10)

        # four options for moving
        random_values = np.random.uniform(0.7, 0.9, 4)
        random_values_2 = np.random.uniform(0, 0.6, 2)
        # attack first
        selected_option[4:8] = 1.0
        selected_option[:4] = random_values
        selected_option[-2:] = random_values_2

        return selected_option


    def handle_step(self, obs, a, reward, next_obs, is_done, timeout, epoch_ended):
        pass

    def handle_epoch(self):
        pass
