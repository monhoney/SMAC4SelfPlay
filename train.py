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

from moon_gym import MoonGym, MoonGymFramework
from util.env import args, device
from util.my_logger import EpochLogger
from util.user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP
from util.user_config import DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
from util.user_config import DEFAULT_BACKEND
from moon_gym import MoonGymFramework
from algo2.register import get_algo, DEFAULT_MAP_SETTINGS

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, 
        datestamp=False, use_wandb=False, run_name=None):

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name, use_wandb=use_wandb,
                         run_name=run_name)
    return logger_kwargs

def run():
    run_name = None
    if args.run_name != "":
        run_name = args.run_name

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, 
        run_name=run_name, use_wandb=args.wandb)

    torch.set_num_threads(torch.get_num_threads())
    assert args.framework == "smac"
    framework = MoonGymFramework.SMAC

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)

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

    for agent_idx in range(args.agent_count):
        logger.setup_pytorch_saver_with_key(\
            algo_dic_list[0][agent_idx].get_model(),
            "%d" % agent_idx)

    # Train 
    start_time = time.time()
    o = env.reset()
    if use_blizzard_ai == True:
        o_dic = {0:o}
    else:
        o_dic = {0:o[0], 1:o[1]}
    ep_ret_dic = {0:0, 1:0}
    ep_len = 0

    for epoch in range(args.epochs):

        if epoch % 100 == 0:
            print ("save replay")
            env.save_replay()

        for t in range(args.steps):
            a_list_dic = {0 : [], 1 : []}

            # Action 결정 및 리워드 받기
            for player_idx in range(2):
                if use_blizzard_ai == True and player_idx == 1:
                    break

                if player_idx == 0:
                    is_train = True
                else:
                    is_train = False

                for agent_idx in range(args.agent_count):
                    a = algo_dic_list[player_idx][agent_idx].act(\
                        o_dic[player_idx][agent_idx], is_train=is_train)

                    a_list_dic[player_idx].append(a)

            if use_blizzard_ai == True:
                next_o, r, d, _ = env.step(a_list_dic[0])
            else:
                next_o, r, d, _ = env.step([a_list_dic[0], a_list_dic[1]])

            if use_blizzard_ai == True:
                ep_ret_dic[0] += r
            else:
                ep_ret_dic[0] += r[0]
                ep_ret_dic[1] += r[1]
            ep_len += 1

            # 알고리즘 Step
            timeout = ep_len == args.eplen
            terminal = d or timeout
            epoch_ended = t == args.steps-1

            for agent_idx in range(args.agent_count):
                if use_blizzard_ai == True:
                    algo_dic_list[0][agent_idx].handle_step(\
                        o_dic[0][agent_idx],
                        a_list_dic[0][agent_idx], r,
                        next_o[agent_idx], d, timeout, epoch_ended)
                else:
                    algo_dic_list[0][agent_idx].handle_step(\
                        o_dic[0][agent_idx],
                        a_list_dic[0][agent_idx], r[0],
                        next_o[0][agent_idx], d, timeout, epoch_ended)
            
            o = next_o
            if use_blizzard_ai == True:
                o_dic = {0:o}
            else:
                o_dic = {0:o[0], 1:o[1]}

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'\
                        % ep_len, flush=True)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret_dic[0], EpLen=ep_len)

                o = env.reset()
                if use_blizzard_ai == True:
                    o_dic = {0:o}
                else:
                    o_dic = {0:o[0], 1:o[1]}
                ep_ret_dic = {0:0, 1:0}
                ep_len = 0

        # Save model
        if (epoch % args.save_freq == 0) or (epoch == args.epochs-1):
            for agent_idx in range(args.agent_count):
                logger.save_state({'env': env}, itr=None, key="%d" % agent_idx)

        # update!
        for agent_idx in range(args.agent_count):
            algo_dic_list[0][agent_idx].handle_epoch()
        
        # Log info about epoch
        logger.log_wandb([['EpRet', True, False],
            ['EpLen', False, True]])

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    logger.finish_wandb()

if __name__ == "__main__":
    run()
