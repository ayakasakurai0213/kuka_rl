import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet as p
from dqn import DQN, get_screen


def main():
    PATH = 'policy_dqn.pt'
    STACK_SIZE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episode = 10
    scores_window = collections.deque(maxlen=100)  # last 100 scores
    env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20, isTest=True)
    env.cid = p.connect(p.DIRECT)
    env.reset()
    # load the model
    checkpoint = torch.load(PATH)

    init_screen = get_screen(env, device=device)
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    policy_net = DQN(STACK_SIZE, screen_height, screen_width, n_actions).to(device)

    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    # evaluate the model
    for i_episode in range(episode):
        env.reset()
        state = get_screen(env, device=device)
        stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
        for t in count():
            stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
            # Select and perform an action
            action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
            _, reward, done, _ = env.step(action.item())
            # Observe new state
            next_state = get_screen(env, device=device)
            stacked_states.append(next_state)
            if done:
                break
        print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")
        
        
if __name__ == "__main__": 
    main()