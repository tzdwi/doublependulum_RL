"""
We're going to be training a Soft Actor-Critic
"""
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .nn_common import this_dir, init_env, device, Transition, ReplayMemory

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

SAC_policy_pickle = this_dir+"/pickles/sac_policy_net.pt"
SAC_Q_pickle = this_dir+"/pickles/sac_q_net.pt"

class SAC_POLICY_DP(nn.Module):
    """
    Our goal is to learn a function pi(s) to output a _distribution_ over 
    actions, conditioned on the state, s. The selected action is then drawn
    from that distribution conditioned on s, in which case our outputs
    are a mean action, and a (log) standard deviation. 
    """
    def __init__(self, n_observations):
        super(DDPG_POLICY_DP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, s)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[mean dF, log sigma dF]...]).
    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        return self.layer3(x)

class SAC_Q1_DP(nn.Module):
    """
    Our goal is to learn a function Q*(s, a) to estimate the reward of
    acting optimally on the state s.
    """
    def __init__(self, n_observations):
        super(DDPG_Q_DP, self).__init__()
        # we want the input to be n_observations+1 to include the action
        self.layer1 = nn.Linear(n_observations+1, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[dF]...]).
    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        return self.layer3(x)

# LEFTOVERS

def get_policy(self, obs):
    if self.do_prob:
        return torch.distributions.Normal(*self.policy_net(obs))
    else:
        return self.policy_net(obs)

def get_action(self, obs):
    if self.do_prob:
        return self.get_policy(obs).sample().item()
    else:
        return self.get_policy(obs).item()
