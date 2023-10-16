import os

class RLAlgoWrapper:
    def __init__(self, env, agent_count, load_model_path):
        self.env = env
        self.agent_count = agent_count
        self.load_model_path = load_model_path

    def get_model(self):
        raise NotImplementedError

    # for inference
    def act(self, obs, is_train):
        raise NotImplementedError

    def handle_step(self, obs, a, reward, next_obs, is_done, timeout, epoch_ended):
        raise NotImplementedError

    def handle_epoch(self): 
        raise NotImplementedError


