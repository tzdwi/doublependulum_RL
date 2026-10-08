"""
We're going to be training a Soft Actor-Critic extension called CrossQ, see https://arxiv.org/pdf/1902.05605
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

from .nn_common import this_dir, plot_dir, vid_dir, init_env, device, Transition, ReplayMemory

import imageio

CQ_policy_pickle = this_dir+"/pickles/crossq_policy_net.pt"
CQ_Q1_pickle = this_dir+"/pickles/crossq_q1_net.pt"
CQ_Q2_pickle = this_dir+"/pickles/crossq_q2_net.pt"

CQ_EPISODE_DURATION_FIG = plot_dir+"/crossq_episode_duration.png"
CQ_CUM_REWARDS_FIG = plot_dir+"/crossq_cum_rewards.png"
CQ_BALANCED_FIG = plot_dir+"/crossq_balanced.png"
CQ_TRUNCATED_FIG = plot_dir+"/crossq_truncated.png"

class CQ_POLICY_DP(nn.Module):
    """
    Our goal is to learn a function pi(s) to output a _distribution_ over 
    actions, conditioned on the state, s. The selected action is then drawn
    from that distribution conditioned on s, in which case our outputs
    are a mean action, and a (log) standard deviation. Note, unlike
    DDPG, we are going to squash the output downstream, so no tanh constraint.
    """
    def __init__(self, n_observations,):
        super(CQ_POLICY_DP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)
        self.activation = nn.Softplus()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[mean dF, log sigma dF]...]).
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)

 
class CQ_Q_DP(nn.Module):
    """
    Our goal is to learn a function Q*(s, a) to estimate the reward of
    acting optimally on the state s.
    """
    def __init__(self, n_observations):
        super(CQ_Q_DP, self).__init__()
        # we want the input to be n_observations+1
        # but the input will be double the batch size
        # so that the net sees both current state and next
        # state
        # We're also training a _wide_ network here!
        self.layer1 = nn.Linear(n_observations+1, 1024)
        # We also add batch normalization between layers
        self.norm1 = nn.BatchNorm1d(
            num_features=1024,
            momentum=0.01, #equiv to 0.99 in CrossQ paper
            eps=0.001
            )
        self.layer2 = nn.Linear(1024, 1024)
        self.norm2 = nn.BatchNorm1d(
            num_features=1024,
            momentum=0.01,
            eps=0.001
            )
        self.layer3 = nn.Linear(1024, 1)
        self.activation = nn.ReLU()

    # Called with either one element to determine next action's reward
    # or a batch during optimization. Returns tensor([[reward]...]).
    def forward(self, x):
        x = self.activation(self.norm1(self.layer1(x)))
        x = self.activation(self.norm2(self.layer2(x)))
        return self.layer3(x)


class CrossQ_Learner:
    def __init__(self, 
                 size=50, 
                 max_F=25.0, 
                 max_ang_vel=4*np.pi, 
                 mm=5.0, 
                 m1=0.5, 
                 l1=10.0, 
                 m2=0.5, 
                 l2=10.0, 
                 dt=0.03, 
                 theta_tol=np.pi/10,
                 batch_size=128,
                 gamma=0.99,
                 alpha=0.2,
                 learning_rate=1e-3,
                 start_steps=256,
                 buffer_length=1e6,
                 policy_delay=3,
                 log_progress=False):

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
    
        self.log_progress=log_progress
        self.log_env=None
        if self.log_progress:
            self.log_env = init_env(size=size,
                            max_F=max_F,
                            max_ang_vel=max_ang_vel,
                            mm=mm,
                            m1=m1,
                            l1=l1,
                            m2=m2,
                            l2=l2,
                            dt=dt,
                            theta_tol=theta_tol,
                            render_mode="rgb_array")

        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor 
        # ALPHA is the regularization term for the entropy
        # LR is the learning rate of the ``AdamW`` optimizer
        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.ALPHA = alpha
        self.LR = learning_rate
        
        # Get the number of state observations
        state, info = self.env.reset()
        self.state = state
        n_observations = len(state)
        self.scale = float(self.env.action_space.high[0])

        # Net to predict the next action
        self.policy_net = CQ_POLICY_DP(n_observations).to(device)
        
        # Nets to predict the reward for the next action, which we take the minimum of
        # otherwise the policy just learns any faults in the one Q net
        self.Q_net_1 = CQ_Q_DP(n_observations).to(device)
        self.Q_net_2 = CQ_Q_DP(n_observations).to(device)
        # NOTE, no target nets in CrossQ!
        
        # Optimizers use LR=10^-3 by default
        # and Beta1=0.5
        self.Q_optimizer = optim.AdamW([{"params":self.Q_net_1.parameters()},
                                       {"params":self.Q_net_2.parameters()}],
                                       lr=self.LR, betas=(0.5, 0.999), amsgrad=True)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), 
                                            lr=self.LR, betas=(0.5, 0.999), amsgrad=True)
        # Huber loss
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(int(buffer_length))
    
        self.steps_done = 0
        self.start_steps = max(start_steps, 2*self.BATCH_SIZE)
        self.policy_delay = policy_delay
        
        self.episode_durations = []
        self.cumulative_rewards = []
        self.final_state_balanced = []
        self.final_state_truncated = []

    def select_action(self, state, do_deterministic=False):
        """
        We sample an action from the policy.

        We also want some exploration, so for the first `start_steps` steps, we instead uniformly sample
        actions from the environment
        """
        if (self.steps_done < self.start_steps) and (not do_deterministic):
            action = torch.as_tensor(self.env.action_space.sample(), device=device, dtype=torch.float32).view(1, 1)
        else:
            with torch.no_grad():
                action_mu, action_logsig = self.policy_net(state).chunk(2, dim=-1)
                if not do_deterministic:
                    action_logsig = action_logsig.clamp(-20, 2)
                    dist = torch.distributions.Normal(action_mu, torch.exp(action_logsig))
                    act = dist.sample()
                else:
                    act = action_mu
                action = self.scale*F.tanh(act)

        return torch.clamp(action, float(self.env.action_space.low[0]), float(self.env.action_space.high[0]))
        
    def optimize_model(self, do_policy_step=False):
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
    
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        next_state_batch = torch.zeros((self.BATCH_SIZE, len(self.state)),device=device)
        if non_final_mask.any():
            next_state_batch[non_final_mask] = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        next_action_batch = torch.zeros(self.BATCH_SIZE, device=device)
        if non_final_mask.any():
            next_action_mus, next_action_logsigmas = self.policy_net(next_state_batch[non_final_mask]).chunk(2, dim=-1)
            next_action_logsigmas = next_action_logsigmas.clamp(-20, 2)
            dists = torch.distributions.Normal(next_action_mus, torch.exp(next_action_logsigmas))
            u = dists.sample()
            next_action_batch[non_final_mask] = self.scale*F.tanh(u).squeeze(-1)
            # see log prob def in appendix C here: https://arxiv.org/pdf/1801.01290
            logprobs_batch = (dists.log_prob(u) - torch.log(1.0-torch.pow(F.tanh(u),2.0)+1e-12)-math.log(self.scale)).squeeze(-1)

        # Compute targets, y
        # Compute Q(s, a)
        # state_batch is shape N_batch x n_observations
        # action_batch is shape N_batch x 1
        # concatenate along final axis
        Q_in = torch.cat((state_batch, action_batch), dim=-1)
        # Same with the input from the next state
        Q_in_next = torch.cat((next_state_batch, next_action_batch.unsqueeze(-1)), dim=-1)
        # Now concatenate along the batch axis
        Q_in = torch.cat((Q_in, Q_in_next), dim=0)

        # Compute Q values
        Q1 = self.Q_net_1(Q_in)
        Q2 = self.Q_net_2(Q_in)

        # Split into current and next values
        Q1, Q1_next = Q1.split(self.BATCH_SIZE, dim=0)
        Q2, Q2_next = Q2.split(self.BATCH_SIZE, dim=0)

        # Compute targets y(r, s', d) = r + gamma*(1-d)*min(Q(s', a')) - alpha log pi(a', s')), a' ~ pi( s')
        # for each batch state according to policy_net and Q_net
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        if non_final_mask.any():
            Q_nexts = torch.cat((Q1_next,Q2_next),dim=-1)
            selected = torch.min(Q_nexts, dim=-1).values
            next_state_values[non_final_mask] = self.GAMMA*(selected[non_final_mask]-self.ALPHA*logprobs_batch)
        
        targets = reward_batch+next_state_values
    
        # Compute Huber loss
        loss = self.criterion(Q1.squeeze(-1), targets.detach()) + self.criterion(Q2.squeeze(-1), targets.detach())
    
        # Gradient descend the Q_net parameters
        self.Q_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q_net_1.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.Q_net_2.parameters(), 100)
        self.Q_optimizer.step()

        if do_policy_step:
            # Now gradient ascend the policy_net parameters
            next_action_mus, next_action_logsigmas = self.policy_net(state_batch).chunk(2, dim=-1)
            next_action_logsigmas = next_action_logsigmas.clamp(-20, 2)
            dists = torch.distributions.Normal(next_action_mus, torch.exp(next_action_logsigmas))
            u = dists.rsample()
            next_action_values = self.scale*F.tanh(u)
            # again, logprob defined in appC here: https://arxiv.org/pdf/1801.01290
            logprobs = (dists.log_prob(u) - torch.log(1.0-torch.pow(F.tanh(u),2.0)+1e-12)-math.log(self.scale)).squeeze(-1)
            Q_in = torch.cat((state_batch, next_action_values), dim=-1)
            # each Q net returns Bx1, so we want to stack along the last dim
            Qs = torch.cat([Q_net(Q_in) for Q_net in [self.Q_net_1, self.Q_net_2]], dim=-1)
            selected = torch.min(Qs, dim=-1).values
            # we want gradient _ascent_ so we use the negative of the sum of action values
            loss = -(selected-self.ALPHA*logprobs).sum()/self.BATCH_SIZE
            self.policy_optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_optimizer.step()
        torch.cuda.empty_cache()

    def run_sim(self, episode_number):
        frames = []
        state, info = self.log_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            # select action based on observed state
            action = self.select_action(state, do_deterministic=True)
            # take step in environment to get next state, reward, and termination/truncation signals
            observation, reward, terminated, truncated, _ = self.log_env.step(action.item())
            # HxWx3
            frame = self.log_env.unwrapped.render(log_reward=True,reward=reward)
            frames.append(frame)
            # done signal is either terminated or truncated
            done = terminated or truncated
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        imageio.mimsave(vid_dir+f"/crossq_ep{episode_number}.mp4", frames, fps=self.log_env.metadata["render_fps"])
            
    
    def train(self, progress=False, make_plots=False, num_episodes = None):
        if not num_episodes:
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                num_episodes = 600
            else:
                num_episodes = 50

        iterator = range(num_episodes)
        if progress:
            iterator = tqdm(iterator)
        
        for i_episode in iterator:
            if i_episode > 0:
                if self.log_env and ((np.log2(i_episode) % 1 == 0) or (i_episode==num_episodes-1)):
                    self.run_sim(i_episode)
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
                    do_policy_step = False
                    if t % self.policy_delay == 0:
                        do_policy_step = True
                    self.optimize_model(do_policy_step=do_policy_step)
        
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
            plt.savefig(CQ_EPISODE_DURATION_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.cumulative_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Rewards')
            plt.savefig(CQ_CUM_REWARDS_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.final_state_truncated)
            plt.xlabel('Episode')
            plt.ylabel('Episode Truncated')
            plt.savefig(CQ_TRUNCATED_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.final_state_balanced)
            plt.xlabel('Episode')
            plt.ylabel('Episode Success')
            plt.savefig(CQ_BALANCED_FIG, bbox_inches='tight')

    def dump(self):
        torch.save(self.policy_net.state_dict(), CQ_policy_pickle)
        torch.save(self.Q_net_1.state_dict(), CQ_Q1_pickle)
        torch.save(self.Q_net_2.state_dict(), CQ_Q2_pickle)

    def load(self):
        self.policy_net.load_state_dict(torch.load(CQ_policy_pickle, weights_only=True, map_location=device))
        self.Q_net_1.load_state_dict(torch.load(CQ_Q1_pickle, weights_only=True, map_location=device))
        self.Q_net_2.load_state_dict(torch.load(CQ_Q2_pickle, weights_only=True, map_location=device))
