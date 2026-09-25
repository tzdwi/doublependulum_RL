"""
We're going to be training a Deep Deterministic Policy Gradient to learn how to control the double pendulum
"""
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .nn_common import this_dir, plot_dir, init_env, device, Transition, ReplayMemory


DDPG_policy_pickle = this_dir+"/pickles/ddpg_policy_net.pt"
DDPG_Q_pickle = this_dir+"/pickles/ddpg_q_net.pt"

DDPG_EPISODE_DURATION_FIG = plot_dir+"/ddpg_episode_duration.png"
DDPG_CUM_REWARDS_FIG = plot_dir+"/ddpg_cum_rewards.png"
DDPG_BALANCED_FIG = plot_dir+"/ddpg_balanced.png"
DDPG_TRUNCATED_FIG = plot_dir+"/ddpg_truncated.png"

class DDPG_POLICY_DP(nn.Module):
    """
    Our goal is to learn a function pi(s) to output an action a as a function
    of a state, s. Alternatively, we can learn a _distribution_, and draw our 
    action from that distribution conditioned on s, in which case our outputs
    are a mean action, and a (log) standard deviation. WERE GOING TO PUT THE DISTRIBUTION VERSION IN A SEPARATE FILE
    """
    def __init__(self, n_observations, scale):
        super(DDPG_POLICY_DP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        self.activation = nn.Softplus()
        self.scale = scale

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[dF]...]).
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.scale*F.tanh(self.layer3(x))

class DDPG_Q_DP(nn.Module):
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
        self.activation = nn.Softplus()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[dF]...]).
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)

class DDPG_Learner:
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
                 tau=0.995,
                 learning_rate=3e-4,
                 start_steps=256,
                 buffer_length=1e6):

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
        # TAU is the update rate of the target network for polyak averaging
        # LR is the learning rate of the ``AdamW`` optimizer
        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = learning_rate
        
        # Get the number of state observations
        state, info = self.env.reset()
        self.state = state
        n_observations = len(state)

        # Net to predict the next action
        self.policy_net = DDPG_POLICY_DP(n_observations, scale=self.env.action_space.high[0]).to(device)
        # target net initialized with same weights
        self.policy_target = DDPG_POLICY_DP(n_observations, scale=self.env.action_space.high[0]).to(device)
        self.policy_target.load_state_dict(self.policy_net.state_dict())
        
        # Net to predict the reward for the next action
        self.Q_net = DDPG_Q_DP(n_observations).to(device)
        # target net initialized with same weights
        self.Q_target = DDPG_Q_DP(n_observations).to(device)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        
        self.Q_optimizer = optim.AdamW(self.Q_net.parameters(),
                                       lr=self.LR, amsgrad=True)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), 
                                            lr=self.LR, amsgrad=True)
        # Huber loss
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(int(buffer_length))
    
        self.steps_done = 0
        self.start_steps = max(start_steps, 2*self.BATCH_SIZE)
        
        self.episode_durations = []
        self.cumulative_rewards = []
        self.final_state_balanced = []
        self.final_state_truncated = []

    def select_action(self, state):
        """
        We deterministically pick an action and add on decaying white noise.

        We also want some exploration, so for the first `start_steps` steps, we uniformly sample
        actions from the environment
        """
        if self.steps_done < self.start_steps:
            action = torch.as_tensor(self.env.action_space.sample(), device=device, dtype=torch.float32).view(1, 1)
        else:
            with torch.no_grad():
                action = self.policy_net(state) 
                steps_ellapsed = self.steps_done-self.start_steps
                eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_ellapsed / self.EPS_DECAY)
                action += eps*torch.randn_like(action)

        return torch.clamp(action, float(self.env.action_space.low[0]), float(self.env.action_space.high[0]))
        
    def optimize_model(self):
        # if our memory is shorter than the batch size, then 
        # we can't sample from it yet, so perform no changes
        if len(self.memory) < self.BATCH_SIZE:
            return
        # Now sample a batch 
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        if not non_final_mask.any():
            non_final_next_states = torch.tensor([])
        else:
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s, a)
        # state_batch is shape N_batch x n_observations
        # action_batch is shape N_batch x 1
        # concatenate along final axis
        Q_in = torch.cat((state_batch, action_batch), dim=-1)
        state_action_values = self.Q_net(Q_in)
    
        # Compute targets y(r, s', d) = r + gamma*(1-d)Q_targ(s', pi(s'))
        # for each batch state according to target policy_net and Q_net
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        if non_final_mask.any():
            with torch.no_grad():
                # we're not updating the targets yet, so don't accumulate grads
                next_action_values = self.policy_target(non_final_next_states)
                # non_final_next_states is shape N_batch_nonfinal x n_observations
                # next action values is shape N_batch_nonfinal x 1
                # concatenate along final axis
                target_input = torch.cat((non_final_next_states, next_action_values), dim=-1)
                next_state_values[non_final_mask] = self.GAMMA*self.Q_target(target_input).squeeze(-1)

        targets = reward_batch+next_state_values
    
        # Compute Huber loss
        loss = self.criterion(state_action_values, targets.unsqueeze(1))
    
        # Gradient descend the Q_net parameters
        self.Q_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q_net.parameters(), 100)
        self.Q_optimizer.step()

        # Now gradient ascend the policy_net parameters
        next_actions = self.policy_net(state_batch)
        Q_in = torch.cat((state_batch, next_actions), dim=-1)
        # we want gradient _ascent_ so we use the negative of the sum of action values
        loss = -self.Q_net(Q_in).sum()/self.BATCH_SIZE
        self.policy_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.policy_optimizer.step()
        
    
    def train(self, progress=False, make_plots=False):

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        iterator = range(num_episodes)
        if progress:
            iterator = tqdm(iterator)
        
        for i_episode in iterator:
            cumulative_reward = 0
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                # select action based on observed state
                action = self.select_action(state)
                # take step in environment to get next state, reward, and termination/truncation signals
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                self.steps_done += 1
                cumulative_reward += reward
                reward = torch.tensor([reward], device=device)
                # done signal is either terminated or truncated
                done = terminated or truncated
        
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
        
                # Move to the next state
                state = next_state

                if len(self.memory) > self.start_steps:
        
                    # Optimization step
                    self.optimize_model()
            
                    # Polyak averaging to update target nets
                    Q_net_state_dict = self.Q_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
    
                    Q_target_state_dict = self.Q_target.state_dict()
                    policy_target_state_dict = self.policy_target.state_dict()
    
                    for key in Q_net_state_dict:
                        Q_target_state_dict[key] = Q_target_state_dict[key]*self.TAU + Q_net_state_dict[key]*(1-self.TAU)
                    for key in policy_net_state_dict:
                        policy_target_state_dict[key] = policy_target_state_dict[key]*self.TAU + policy_net_state_dict[key]*(1-self.TAU)
                    
                    self.Q_target.load_state_dict(Q_target_state_dict)
                    self.policy_target.load_state_dict(policy_target_state_dict)
        
                if done:
                    self.episode_durations.append(t + 1)
                    self.cumulative_rewards.append(cumulative_reward)
                    if reward == 1000:
                        self.final_state_balanced.append(1)
                    else:
                        self.final_state_balanced.append(0)
                    if truncated:
                        self.final_state_truncated.append(1)
                    else:
                        self.final_state_truncated.append(0)
                    break
        if make_plots:
            fig = plt.figure(dpi=300)
            plt.plot(np.arange(num_episodes), self.episode_durations)
            plt.xlabel('Episode')
            plt.ylabel('Episode Duration')
            plt.savefig(DDPG_EPISODE_DURATION_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.cumulative_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Rewards')
            plt.savefig(DDPG_CUM_REWARDS_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.final_state_truncated)
            plt.xlabel('Episode')
            plt.ylabel('Episode Truncated')
            plt.savefig(DDPG_TRUNCATED_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.final_state_balanced)
            plt.xlabel('Episode')
            plt.ylabel('Episode Success')
            plt.savefig(DDPG_BALANCED_FIG, bbox_inches='tight')

    def dump(self):
        torch.save(self.policy_target.state_dict(), DDPG_policy_pickle)
        torch.save(self.Q_target.state_dict(), DDPG_Q_pickle)

    def load(self):
        self.policy_target.load_state_dict(torch.load(DDPG_policy_pickle, weights_only=True, map_location=device))
        self.Q_target.load_state_dict(torch.load(DDPG_Q_pickle, weights_only=True, map_location=device))

        self.policy_net.load_state_dict(self.policy_target.state_dict())
        self.Q_net.load_state_dict(self.Q_target.state_dict())


