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
from algo2.register import get_algo, DEFAULT_MAP_SETTINGS

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
    if args.enemy_algo.upper() == 'BLIZZARD':
        use_blizzard_ai = True
    else:
        use_blizzard_ai = False
        enemy_algo = get_algo(args.enemy_algo)

    # set map
    if args.env == "":
        args.env = DEFAULT_MAP_SETTINGS[(use_blizzard_ai, args.agent_count)]
        print ("Map : ", args.env)

    # ENV
    if use_blizzard_ai == True:
        env = MoonGym(args.env, framework=framework,
            selfplay=False)
    else:
        env = MoonGym(args.env, framework=framework,
            selfplay=True)

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
    if use_blizzard_ai == False:
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
    dead_players = []
    dead_enemies = []

    for round in tqdm.tqdm(range(args.test_round_count)):
        o = env.reset()
        if use_blizzard_ai == True:
            o_dic = {0:o}
        else:
            o_dic = {0:o[0], 1:o[1]}
        eplen = 0

        for t in range(args.eplen):
            a_list_dic = {0 : [], 1 : []}

            # Action 결정 및 리워드 받기
            for player_idx in range(2):
                if use_blizzard_ai == True and player_idx == 1:
                    break

                for agent_idx in range(args.agent_count):
                    a = algo_dic_list[player_idx][agent_idx].act(\
                        o_dic[player_idx][agent_idx], is_train=False)

                    a_list_dic[player_idx].append(a)

            if use_blizzard_ai == True:
                next_o, r, d, info = env.step(a_list_dic[0])
            else:
                next_o, r, d, info = env.step([a_list_dic[0], a_list_dic[1]])
                
            # 알고리즘 Step
            if d == True:
                if use_blizzard_ai == True:
                    battle_won = info['battle_won']
                    dead_player = info['dead_allies']
                    dead_enemy = info['dead_enemies']
                else:
                    battle_won = info[0]['battle_won']
                    dead_player = info[0]['dead_allies']
                    dead_enemy = info[0]['dead_enemies']

                is_win = False
                if t == args.eplen -1:
                    draw = draw + 1
                elif battle_won == True:
                    player_win = player_win + 1
                    is_win = True
                else:
                    enemy_win = enemy_win + 1
                    assert info[0]['dead_allies'] == args.agent_count

                dead_players.append(dead_player)
                dead_enemies.append(dead_enemy)

                eplen = t + 1
                break
            
            o = next_o

        eplens.append(eplen)
        if is_win == True:
            win_eplens.append(eplen)

    total_point = player_win * 2.0 + draw * 1.0 - \
        sum(dead_players) / args.agent_count + \
        sum(dead_enemies) / args.agent_count

    print ("*" * 80)
    print ("%s vs %s" % (args.player_algo, args.enemy_algo))
    print ("Player Win : %d" % player_win)
    print ("Enemy Win : %d" % enemy_win)
    print ("Draw : %d" % draw)
    print ("Player Win Ratio : %.3f" % (player_win / args.test_round_count))
    print ("Average EpLen : %.3f" % (np.array(eplens).mean()))
    if player_win > 0:
        print ("Average Win EpLen : %.3f" % (np.array(win_eplens).mean()))
    print ("Average Dead Player : %.3f" % (np.array(dead_players).mean()))
    print ("Average Dead Enemy : %.3f" % (np.array(dead_enemies).mean()))
    print ("Point :%.3f" % total_point)
    print ("*" * 80)

    if args.json_output != "":
        result = {"PlayerWin" : player_win,
            "EnemyWin" : enemy_win,
            "Draw" : draw,
            "EpLens" : eplens,
            "WinEpLens" : win_eplens,
            "DeadPlayers" : dead_players,
            "DeadEnemies" : dead_enemies}
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=4) 

