"""
Some common tools across the RL algorithms I want to play with
"""

import gymnasium as gym
import torch
import random
from collections import namedtuple, deque
import os
import numpy as np
from pathlib import Path

from .. import max_seconds

this_dir = str(os.path.dirname(os.path.realpath(__file__)))
plot_dir = str(Path(__file__).parent.parent.parent)+"/scripts/figures"
vid_dir = str(Path(__file__).parent.parent.parent)+"/scripts/videos"

def init_env(size=50, max_F=25.0, max_ang_vel=4*np.pi, mm=5.0, m1=0.5, l1=10.0, m2=0.5, l2=10.0, dt=0.03, theta_tol=np.pi/10, render_mode=None):
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
                   theta_tol=theta_tol,
                   render_mode=render_mode
                  )
    return env

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# define transitions from state to state as a function of action
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TransitionSequence = namedtuple('TransitionSequence',
                        ('state', 'action', 'reward'))

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

class ExperienceMemory(object):
    def __init__(self, capacity, sequence_length=80, sequence_overlap=40):
            self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a sequence of transitions"""
        self.memory.append([TransitionSequence(*a) for a in args])

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
