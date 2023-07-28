from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import os.path as osp, time, atexit, os
import sys
if 'BASE_DIR' in os.environ:
    sys.path.append(os.environ['BASE_DIR'])
else:
    sys.path.append(os.path.join(os.environ['HOME'], 'work', 'SMAC2Study'))
import algo.mppo.core as core
from util.my_logger import EpochLogger

from util.user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP
from util.user_config import DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH, DEFAULT_BACKEND

from functools import reduce

from moon_gym import MoonGym, MoonGymFramework
from pathlib import Path
import pprint
import datetime
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Simple64_Tank')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='SMAC_SP')
parser.add_argument('--framework', default="smac", choices=['smac'])
parser.add_argument('--eplen', type=int, default=100)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--selfplay', action='store_true')
parser.add_argument('--no-selfplay', dest='selfplay', action='store_false')
parser.set_defaults(selfplay=True)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--no-wandb', dest='wandb', action='store_false')
parser.set_defaults(wandb=False)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.set_defaults(train=True)
parser.add_argument('--model_filepath', type=str, default="")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
#       adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
#       self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}



def mppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=100,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    obs_flat_dim = (reduce (lambda x,y:x*y, env.observation_space.shape),)
    act_dim = env.action_space.shape
    n_agents = env.n_agents

    # Create actor-critic module
    ac_list_dic = {0 : [], 1 : []}
    for p_id in range(2):
        for _ in range(n_agents):
            ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
            ac_list_dic[p_id].append(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac_list_dic[0][0].pi, 
            ac_list_dic[0][0].v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch)
    buf_list_dic = {0 : [], 1 : []}
    for p_id in range(2):
        for _ in range(n_agents):
            buf = PPOBuffer(obs_flat_dim, act_dim, local_steps_per_epoch, gamma, lam)
            buf_list_dic[p_id].append(buf)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, player_idx, agent_idx):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac_list_dic[player_idx][agent_idx].pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, player_idx, agent_idx):
        obs, ret = data['obs'], data['ret']
        return ((ac_list_dic[player_idx][agent_idx].v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer_list_dic = {0 : [], 1 : []}
    vf_optimizer_list_dic = {0 : [], 1 : []}

    for player_idx in range(2):
        for agent_idx in range(n_agents):
            pi_optimizer = Adam(ac_list_dic[player_idx][agent_idx].pi.parameters(), lr=pi_lr)
            vf_optimizer = Adam(ac_list_dic[player_idx][agent_idx].v.parameters(), lr=vf_lr)
            pi_optimizer_list_dic[player_idx].append(pi_optimizer)
            vf_optimizer_list_dic[player_idx].append(vf_optimizer)

    # Set up model saving
    # XXX: check!!!
    logger.setup_pytorch_saver(ac_list_dic)

    def update():

        pi_l_old_list_dic = {0:[],1:[]}
        v_l_old_list_dic = {0:[],1:[]}
        kl_list_dic = {0:[],1:[]}
        ent_list_dic = {0:[],1:[]}
        cf_list_dic = {0:[],1:[]}
        delta_loss_pi_list_dic = {0:[],1:[]}
        delta_loss_v_list_dic = {0:[],1:[]}

        for player_idx in range(2):
            for agent_idx in range(n_agents):
                data = buf_list_dic[player_idx][agent_idx].get()

                pi_l_old, pi_info_old = compute_loss_pi(data, player_idx, agent_idx)
                pi_l_old = pi_l_old.item()
                v_l_old = compute_loss_v(data, player_idx, agent_idx).item()

                # Train policy with multiple steps of gradient descent
                for i in range(train_pi_iters):
                    pi_optimizer_list_dic[player_idx][agent_idx].zero_grad()
                    loss_pi, pi_info = compute_loss_pi(data, player_idx, agent_idx)
                    kl = pi_info['kl']
                    if kl > 1.5 * target_kl:
                        logger.log('Early stopping at step %d due to reaching max kl.'%i)
                        break
                    loss_pi.backward()
                    pi_optimizer_list_dic[player_idx][agent_idx].step()

                # XXX: do not log...
                #logger.store(StopIter=i)

                # Value function learning
                for i in range(train_v_iters):
                    vf_optimizer_list_dic[player_idx][agent_idx].zero_grad()
                    loss_v = compute_loss_v(data, player_idx, agent_idx)
                    loss_v.backward()
                    vf_optimizer_list_dic[player_idx][agent_idx].step()

                # Log changes from update
                kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

                pi_l_old_list_dic[player_idx].append(pi_l_old)
                v_l_old_list_dic[player_idx].append(v_l_old)
                kl_list_dic[player_idx].append(kl)
                ent_list_dic[player_idx].append(ent)
                cf_list_dic[player_idx].append(cf)
                delta_loss_pi_list_dic[player_idx].append(loss_pi.item() - pi_l_old)
                delta_loss_v_list_dic[player_idx].append(loss_v.item() - v_l_old)

            if player_idx == 0: 
                logger.store(P0LossPi=np.mean(pi_l_old_list_dic[player_idx]), 
                        P0LossV=np.mean(v_l_old_list_dic[player_idx]),
                        P0KL=np.mean(kl_list_dic[player_idx]),
                        P0Entropy=np.mean(ent_list_dic[player_idx]),
                        P0ClipFrac=np.mean(cf_list_dic[player_idx]),
                        P0DeltaLossPi=np.mean(delta_loss_pi_list_dic[player_idx]),
                        P0DeltaLossV=np.mean(delta_loss_v_list_dic[player_idx]))
            else:
                logger.store(P1LossPi=np.mean(pi_l_old_list_dic[player_idx]), 
                        P1LossV=np.mean(v_l_old_list_dic[player_idx]),
                        P1KL=np.mean(kl_list_dic[player_idx]),
                        P1Entropy=np.mean(ent_list_dic[player_idx]),
                        P1ClipFrac=np.mean(cf_list_dic[player_idx]),
                        P1DeltaLossPi=np.mean(delta_loss_pi_list_dic[player_idx]),
                        P1DeltaLossV=np.mean(delta_loss_v_list_dic[player_idx]))

    # Prepare for interaction with environment
    start_time = time.time()

    o = env.reset()
    o_dic = {0:o[0], 1:o[1]}
    ep_ret_dic = {0:0, 1:0}
    ep_len = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        if epoch % 100 == 0:
            print ("save replay")
            env.save_replay()

        for t in range(local_steps_per_epoch):
            a_list_dic = {0 : [], 1 : []}
            v_list_dic = {0 : [], 1 : []}
            logp_list_dic = {0 : [], 1 : []}

            for player_idx in range(2):
                for agent_idx in range(n_agents):
                    a, v, logp = \
                        ac_list_dic[player_idx][agent_idx].step(torch.as_tensor(\
                            o_dic[player_idx][agent_idx], 
                            dtype=torch.float32).to(device))
                    
                    a_list_dic[player_idx].append(a)
                    v_list_dic[player_idx].append(v)
                    logp_list_dic[player_idx].append(logp)

            next_o, r, d = env.step([a_list_dic[0], a_list_dic[1]])
            ep_ret_dic[0] += r[0]
            ep_ret_dic[1] += r[1]
            ep_len += 1

            # save and log
            for player_idx in range(2):
                for agent_idx in range(n_agents):
                    buf_list_dic[player_idx][agent_idx].store(\
                        o_dic[player_idx][agent_idx], 
                        a_list_dic[player_idx][agent_idx], 
                        r[player_idx], 
                        v_list_dic[player_idx][agent_idx], 
                        logp_list_dic[player_idx][agent_idx])

                if player_idx == 0:
                    logger.store(P0VVals=np.mean(v_list_dic[0]))
                else:
                    logger.store(P1VVals=np.mean(v_list_dic[1]))
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                for player_idx in range(2):
                    for agent_idx in range(n_agents):
                        if epoch_ended and not(terminal) and agent_idx == 0 and player_idx == 0:
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            _, v, _ = ac_list_dic[player_idx][agent_idx].step(\
                                torch.as_tensor(o_dic[player_idx][agent_idx], 
                                    dtype=torch.float32).to(device))
                        else:
                            v = 0
                        buf_list_dic[player_idx][agent_idx].finish_path(v)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(P0EpRet=ep_ret_dic[0], P1EpRet=ep_ret_dic[1], EpLen=ep_len)
                o = env.reset()
                ep_ret_dic = {0:0, 1:0}
                ep_len = 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()
        
        # Log info about epoch
        logger.log_wandb([['P0EpRet', True, False],
            ['P1EpRet', True, False],
            ['EpLen', False, True],
            ['P0VVals', True, False],
            ['P1VVals', True, False],
            ['P0LossPi', False, True],
            ['P1LossPi', False, True],
            ['P0LossV', False, True],
            ['P1LossV', False, True],
            ['P0DeltaLossPi', False, True],
            ['P1DeltaLossPi', False, True],
            ['P0DeltaLossV', False, True],
            ['P1DeltaLossV', False, True],
            ['P0Entropy', False, True],
            ['P1Entropy', False, True],
            ['P0KL', False, True],
            ['P1KL', False, True],
            ['P0ClipFrac', False, True],
            ['P1ClipFrac', False, True]])

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('P0EpRet', with_min_and_max=True)
        logger.log_tabular('P1EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('P0VVals', with_min_and_max=True)
        logger.log_tabular('P1VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('P0LossPi', average_only=True)
        logger.log_tabular('P1LossPi', average_only=True)
        logger.log_tabular('P0LossV', average_only=True)
        logger.log_tabular('P1LossV', average_only=True)
        logger.log_tabular('P0DeltaLossPi', average_only=True)
        logger.log_tabular('P1DeltaLossPi', average_only=True)
        logger.log_tabular('P0DeltaLossV', average_only=True)
        logger.log_tabular('P1DeltaLossV', average_only=True)
        logger.log_tabular('P0Entropy', average_only=True)
        logger.log_tabular('P1Entropy', average_only=True)
        logger.log_tabular('P0KL', average_only=True)
        logger.log_tabular('P1KL', average_only=True)
        logger.log_tabular('P0ClipFrac', average_only=True)
        logger.log_tabular('P1ClipFrac', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    logger.finish_wandb()

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False, use_wandb=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false, 

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in 
    ``spinup/user_config.py``. 

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

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
                         exp_name=exp_name, use_wandb=use_wandb)
    return logger_kwargs

if __name__ == '__main__':

    if args.train == True:
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, use_wandb=args.wandb)
        torch.set_num_threads(torch.get_num_threads())

        assert args.framework == "smac"
        framework = MoonGymFramework.SMAC

        mppo(lambda : MoonGym(args.env, framework=framework,
            selfplay=args.selfplay),
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            max_ep_len=args.eplen, logger_kwargs=logger_kwargs)

    else:
        #args.model_filepath = "/Users/moonhoenlee/work/SMAC2Study/data/SMAC_8m_mppo/SMAC_8m_mppo_s0/pyt_save/model.pt"
        if args.model_filepath == "":
            print ("You should set model_filepath in the test mode")
            sys.exit(1)

        actor_critic=core.MLPActorCritic
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)

        loaded_model_data = torch.load(args.model_filepath, map_location=device)

        framework = MoonGymFramework.SMAC
        env = MoonGym(args.env, framework=framework, selfplay=args.selfplay)

        obs_dim = env.observation_space.shape
        obs_flat_dim = (reduce (lambda x,y:x*y, env.observation_space.shape),)
        act_dim = env.action_space.shape
        n_agents = env.n_agents

        ac_list_dic = {0 : [], 1 : []}
        for p_id in range(2):
            for a_id in range(n_agents):
                #ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
                #ac.load_state_dict(loaded_model_data[p_id][a_id])
                ac = loaded_model_data[p_id][a_id]
                ac.eval()
                ac_list_dic[p_id].append(ac)

        for test_idx in range(5):
            o = env.reset()
            o_dic = {0:o[0], 1:o[1]}
            ep_ret_dic = {0:0, 1:0}
            ep_len = 0

            for t in range(args.eplen):
                a_list_dic = {0 : [], 1 : []}
                for player_idx in range(2):
                    for agent_idx in range(n_agents):
                        a = ac_list_dic[player_idx][agent_idx].step_max(torch.as_tensor(\
                                o_dic[player_idx][agent_idx], 
                                dtype=torch.float32).to(device))
                        a_list_dic[player_idx].append(a)
                next_o, r, d = env.step([a_list_dic[0], a_list_dic[1]])
                ep_ret_dic[0] += r[0]
                ep_ret_dic[1] += r[1]
                ep_len += 1

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == args.eplen
                terminal = d or timeout

                if terminal or timeout:
                    print ("[TEST Episode #%d] EPLEN=%d, REWARD:(%.2f,%.2f)" %\
                        (test_idx, ep_len, ep_ret_dic[0], ep_ret_dic[1]))
                    break

