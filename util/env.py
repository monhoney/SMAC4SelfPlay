import os
import sys
import torch
if 'BASE_DIR' in os.environ:
    sys.path.append(os.environ['BASE_DIR'])
else:
    sys.path.append(os.path.join(os.environ['HOME'], 'work', 'SMAC2Study'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Simple64_Tank_v2')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--exp_name', type=str, default='selfplay')
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
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--agent_count', type=int, default=4)
parser.add_argument('--player_algo', type=str, default='PPO')
parser.add_argument('--enemy_algo', type=str, default='RANDOM')
parser.add_argument('--player_model_path', type=str, default='')
parser.add_argument('--enemy_model_path', type=str, default='')
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--test_round_count', type=int, default=20)
parser.add_argument('--json_output', type=str, default='')
########## SAC arguments ###################
parser.add_argument('--hard_update', action='store_true')
parser.add_argument('--no-hard_update', dest='hard_update', action='store_false')
parser.set_defaults(hard_update=False)
parser.add_argument('--policy_type', type=str, default='Gaussian', 
                    help='Deterministic can be another option')
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--sac_lr', type=float, default=0.0003)
parser.add_argument('--automatic_entropy_tuning', action='store_true')
parser.add_argument('--no-automatic_entropy_tuning', dest='automatic_entropy_tuning', action='store_false')
parser.set_defaults(automatic_entropy_tuning=False)
################################################
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
