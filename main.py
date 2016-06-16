import dqn
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env_name', type=str, help='an integer for the accumulator')
args = parser.parse_args()

env_name = args.env_name
if env_name is None:
    env_name = "Breakout-v0"

dqn = dqn.DQN(env_name)
dqn.train()
