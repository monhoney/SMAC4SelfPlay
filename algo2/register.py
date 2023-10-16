#########################################
# Algorithm List
#########################################
from algo2.ppo.run      import PPO
from algo2.random.run   import RANDOM

ALGO_DICT = {
    "PPO"       : PPO,
    "RANDOM"    : RANDOM
}

def get_algo(algo_name):
    assert algo_name.upper() in ALGO_DICT
    return ALGO_DICT[algo_name.upper()]
    
