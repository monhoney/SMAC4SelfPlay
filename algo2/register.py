#########################################
# Algorithm List
#########################################
from algo2.ppo.run      import PPO
from algo2.random.run   import RANDOM
from algo2.sac.run      import SAC
from algo2.ddpg2.run     import DDPG
from algo2.computer.run     import COMPUTER


ALGO_DICT = {
    "PPO"       : PPO,
    "RANDOM"    : RANDOM,
    "SAC"       : SAC,
    "DDPG"      : DDPG,
    "COMPUTER"  : COMPUTER
}

DEFAULT_MAP_SETTINGS = {
    (True, 1) : "Simple64_1Tank_v2_ai",
    (True, 2) : "Simple64_2Tank_v2_ai",
    (True, 4) : "Simple64_Tank_v2_ai",
    (True, 8) : "Simple64_8Tank_v2_ai",
    (False, 1) : "Simple64_1Tank_v2",
    (False, 2) : "Simple64_2Tank_v2",
    (False, 4) : "Simple64_Tank_v2",
    (False, 8) : "Simple64_8Tank_v2"
}


def get_algo(algo_name):
    assert algo_name.upper() in ALGO_DICT
    return ALGO_DICT[algo_name.upper()]
    
