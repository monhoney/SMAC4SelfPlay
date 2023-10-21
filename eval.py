import os
import torch
from pathlib import Path
import time
import os.path as osp, time, atexit, os
import itertools
from copy import deepcopy
import numpy as np
from functools import reduce
import pprint
import datetime
import argparse
import tqdm
import json

from moon_gym import MoonGym, MoonGymFramework
from util.env import args, device
from util.my_logger import EpochLogger
from util.user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP
from util.user_config import DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
from util.user_config import DEFAULT_BACKEND
from moon_gym import MoonGymFramework
from algo2.register import get_algo

if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())
    assert args.framework == "smac"
    framework = MoonGymFramework.SMAC

    # Random seed
    seed = args.seed
    seed += 10000 * 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ALG
    player_algo = get_algo(args.player_algo)
    enemy_algo = get_algo(args.enemy_algo)

    # ENV
    env = MoonGym(args.env, framework=framework,
        selfplay=args.selfplay)

    n_agents = env.n_agents
    assert n_agents == args.agent_count

    algo_dic_list = dict()

    # Agent Player - #0
    p_id = 0
    algo_dic_list[p_id] = []
    for agent_idx in range(args.agent_count):
        model_filepath = ""
        if len(args.player_model_path) > 0:
            model_filepath = os.path.join(args.player_model_path, 
                "%d/model.pt" % agent_idx)
        algo_dic_list[p_id].append(player_algo(env, args.agent_count, 
            model_filepath))

    # Agent Enemy - #1
    p_id = 1
    algo_dic_list[p_id] = []
    for agent_idx in range(args.agent_count):
        model_filepath = ""
        if len(args.enemy_model_path) > 0:
            model_filepath = os.path.join(args.enemy_model_path, 
                "%d/model.pt" % agent_idx)
        algo_dic_list[p_id].append(enemy_algo(env, args.agent_count, 
            model_filepath))


    player_win = 0
    enemy_win = 0
    draw = 0
    eplens = []
    win_eplens = []
    player_rewards = []
    enemy_rewards = []

    for round in tqdm.tqdm(range(args.test_round_count)):
        player_reward = 0
        enemy_reward = 0
        o = env.reset()
        o_dic = {0:o[0], 1:o[1]}
        eplen = 0

        for t in range(args.eplen * 3):
            a_list_dic = {0 : [], 1 : []}

            # Action 결정 및 리워드 받기
            for player_idx in range(2):
                for agent_idx in range(args.agent_count):
                    a = algo_dic_list[player_idx][agent_idx].act(\
                        o_dic[player_idx][agent_idx], is_train=False)

                    a_list_dic[player_idx].append(a)

            next_o, r, d = env.step([a_list_dic[0], a_list_dic[1]])
            player_reward += r[0]
            enemy_reward += r[1]

            # 알고리즘 Step
#           if d == True:
#               if t == args.eplen -1:
#                   draw = draw + 1
#               elif player_reward > enemy_reward:
#                   player_win = player_win + 1
#               else:
#                   enemy_win = enemy_win + 1
#               eplen = t + 1
#               break
                
            if d == True:
                is_win = False
                if t == args.eplen -1:
                    draw = draw + 1
                elif player_reward > enemy_reward:
                    player_win = player_win + 1
                    is_win = True
                else:
                    enemy_win = enemy_win + 1
                eplen = t + 1
                break
            
            o = next_o

        player_rewards.append(player_reward)
        enemy_rewards.append(enemy_reward)

        eplens.append(eplen)
        if is_win == True:
            win_eplens.append(eplen)

    print ("*" * 80)
    print ("%s vs %s" % (args.player_algo, args.enemy_algo))
    print ("Player Win : %d" % player_win)
    print ("Enemy Win : %d" % enemy_win)
    print ("Draw : %d" % draw)
    print ("Player Win Ratio : %.3f" % (player_win / args.test_round_count))
    print ("Average Player Return : %.3f" % (np.array(player_rewards).mean()))
    print ("Average Enemy Return : %.3f" % (np.array(enemy_rewards).mean()))
    print ("Average EpLen : %.3f" % (np.array(eplens).mean()))
    if player_win > 0:
        print ("Average Win EpLen : %.3f" % (np.array(win_eplens).mean()))
    print ("*" * 80)

    if args.json_output != "":
        result = {"PlayerWin" : player_win,
            "EnemyWin" : enemy_win,
            "Draw" : draw,
            "PlayerReturns" : player_rewards,
            "EnemyReward" : enemy_rewards,
            "EpLens" : eplens}
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=4) 

