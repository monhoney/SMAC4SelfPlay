#########################################
# Algorithm List
#########################################
from algo2.ppo.run      import PPO
from algo2.random.run   import RANDOM
from algo2.sac.run      import SAC
from algo2.ddpg2.run     import DDPG

ALGO_DICT = {
    "PPO"       : PPO,
    "RANDOM"    : RANDOM,
    "SAC"       : SAC,
    "DDPG"      : DDPG
}

def get_algo(algo_name):
    assert algo_name.upper() in ALGO_DICT
    return ALGO_DICT[algo_name.upper()]
    
