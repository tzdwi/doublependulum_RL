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

def init_env(size=50, max_F=25.0, max_ang_vel=4*np.pi, mm=5.0, m1=0.5, l1=10.0, m2=0.5, l2=10.0, dt=0.03, theta_tol=np.pi/10, action_repeat=3, render_mode=None):
    max_episode_steps = int(max_seconds / (dt * action_repeat))
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
                   action_repeat=action_repeat,
                   render_mode=render_mode
                  )
    return env

def warmup_action(action_space):
    scale = float(action_space.high[0])
    action = np.random.normal(0.0, 0.2 * scale)
    return float(np.clip(action, action_space.low[0], action_space.high[0]))

class EarlyStop:
    """Stop training once the task is solved, or once episode length stops improving."""

    def __init__(self, window=100, solved_count=95, patience=15000, min_gain=2):
        self.window = window
        self.solved_count = solved_count
        self.patience = patience
        self.min_gain = min_gain
        self.best = None
        self.since = 0
        self.hold = 0
        self.improved = False

    def reset(self):
        # Keep the all-time best so a post-reset dip cannot replace its checkpoint.
        # Restart the stall timer, and ignore windows that still contain pre-reset episodes.
        self.since = 0
        self.hold = self.window
        self.improved = False

    def check(self, durations, successes, actor_started):
        self.improved = False
        if len(durations) < self.window or not actor_started:
            return None
        if self.hold:
            self.hold -= 1
            return None
        recent = durations[-self.window:]
        recent_len = float(np.mean(recent))
        recent_succ = int(sum(successes[-self.window:]))
        if recent_succ >= self.solved_count:
            return f"solved: {recent_succ}/{self.window} of the last episodes succeeded"
        if self.best is None or recent_len > self.best + self.min_gain:
            self.best = recent_len
            self.since = 0
            self.improved = True
        else:
            self.since += 1
            if self.since >= self.patience:
                return (f"stalled: mean length {recent_len:.0f} has not beaten {self.best:.0f} "
                        f"by {self.min_gain} for {self.patience} episodes")
        return None

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
