"""
We're going to be training a Soft Actor-Critic
"""

import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .. import max_seconds
import DoublePendulum

def init_env(size=5, max_F=5, max_ang_vel=4*np.pi, mm=5.0, m1=1.0, l1=1.0, m2=1.0, l2=1.0, dt=0.03, theta_tol=np.pi/10):
    max_episode_steps = int(max_seconds / dt)
    env = gym.make("DoublePendulum/DoublePendulum-v0",
                   max_episode_steps=max_episode_steps,
                   size=size,
                   max_F=max_F,
                   max_ang_vel=max_ang_vel,
                   mm=mm,
                   m1=m1,
                   l1=l1,
                   m2=m2,
                   l2=l2,
                   dt=dt,
                   theta_tol=theta_tol
                  )
    return env

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

# define transitions from state to state as a function of action
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# hold transitions in memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
