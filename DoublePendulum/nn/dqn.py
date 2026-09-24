"""
We're going to be training a Deep Q-Network to learn action policy
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
from .. import DoublePendulum

def init_env(size=5, max_F=5, max_ang_vel=4*np.pi, mm=5.0, m1=1.0, l1=1.0, m2=1.0, l2=1.0, dt=0.03, theta_tol=np.pi/10):
    max_episode_steps = int(max_seconds / dt)
    env = gym.make("DoublePendulum/DoublePendulum-v0",
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

class DQN_DP(nn.Module):

    def __init__(self, n_observations, n_actions=1):
        super(DQN_DP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[dF]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_Learner:
    def __init__(self, 
                 size=5, 
                 max_F=5, 
                 max_ang_vel=4*np.pi, 
                 mm=5.0, 
                 m1=1.0, 
                 l1=1.0, 
                 m2=1.0, 
                 l2=1.0, 
                 dt=0.03, 
                 theta_tol=np.pi/10,
                 batch_size=128,
                 gamma=0.99,
                 eps_start=0.9,
                 eps_end=0.01,
                 eps_decay=2500,
                 tau=0.005,
                 learning_rate=3e-4,
                 buffer_length=10000):

    self.env = init_env(size=size,
                        max_F=max_F,
                        max_ang_vel=max_ang_vel,
                        mm=mm,
                        m1=m1,
                        l1=l1,
                        m2=m2,
                        l2=l2,
                        dt=dt,
                        theta_tol=theta_tol)

    

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    
    self.BATCH_SIZE = batch_size
    self.GAMMA = gamma
    self.EPS_START = eps_start
    self.EPS_END = eps_end
    self.EPS_DECAY = eps_decay
    self.TAU = tau
    self.LR = learning_rate


    # Get number of actions from gym action space
    self.n_actions = self.env.action_space.n
    # Get the number of state observations
    state, info = self.env.reset()
    self.state = state
    n_observations = len(state)

    # Net to predict the next action
    self.policy_net = DQN(n_observations, n_actions).to(device)
    # Net to predict the reward for the next action
    self.target_net = DQN(n_observations, n_actions).to(device)
    self.target_net.load_state_dict(policy_net.state_dict())
    
    self.optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)
    # Huber loss
    self.criterion = nn.SmoothL1Loss()
    self.memory = ReplayMemory(buffer_length)

    self.steps_done = 0
    
    self.episode_durations = []

    def select_action(self, state):
        """
        We have an exponentially decaying probability of picking a
        random action vs. and informed action from our policy network
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # the policy net directly predicts
                # the next action given the state
                return self.policy_net(state)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.float32)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t). These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
    
        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50
        
        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
        
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
        
                # Move to the next state
                state = next_state
        
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
        
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
        
                if done:
                    self.episode_durations.append(t + 1)
                    break


