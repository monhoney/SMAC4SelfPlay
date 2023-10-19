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
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
