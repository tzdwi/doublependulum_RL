"""
We're going to be training a giant model for funsies:
BRO (Better, Regularized, Optimized), see https://arxiv.org/pdf/2405.16158

BRO uses quantile regression to estimate the distribution of returns.
More info in:
https://arxiv.org/pdf/1806.06923
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

BRO_optimistic_pickle = this_dir+"/pickles/bro_optimistic_net.pt"
BRO_pessimistic_pickle = this_dir+"/pickles/bro_pessimistic_net.pt"
BRO_Q1_pickle = this_dir+"/pickles/bro_q1_net.pt"
BRO_Q2_pickle = this_dir+"/pickles/bro_q2_net.pt"

BRO_EPISODE_DURATION_FIG = plot_dir+"/bro_episode_duration.png"
BRO_CUM_REWARDS_FIG = plot_dir+"/bro_cum_rewards.png"
BRO_BALANCED_FIG = plot_dir+"/bro_balanced.png"
BRO_TRUNCATED_FIG = plot_dir+"/bro_truncated.png"

class BRO_Block(nn.Module):
    """
    A generic block to be used in BroNet
    """
    def __init__(self, dim_in=256, dim_hid=256, activation='relu'):
        super(BRO_Block, self).__init__()
        self.dense1 = nn.Linear(dim_in, dim_hid)
        self.norm1 = nn.LayerNorm(dim_hid)
        self.dense2 = nn.Linear(dim_hid, dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        out = self.dense1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dense2(out)
        out = self.norm2(out)
        return out + x #residual connection

class BroNet(nn.Module):
    """
    The actual Bro Net!
    """
    def __init__(self, dim_in=8, dim_out=100, dim_block=256, dim_hid=256, N_block=2, activation='relu'):
        super(BroNet, self).__init__()
        self.dense_in = nn.Linear(dim_in, dim_block)
        self.norm_in = nn.LayerNorm(dim_block)
        self.blocks = [BRO_Block(dim_in=dim_block, dim_hid=dim_hid, activation=activation) for i in range(N_block)]
        if activation == 'relu':
            self.activation = nn.ReLU()
        self.dense_out = nn.Linear(dim_block, dim_out)
        self.layers = nn.ModuleList([self.dense_in, self.norm_in, self.activation, *self.blocks, self.dense_out])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BRO_Learner:
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
                 replay_ratio=10,
                 dim_block=256,
                 dim_hid=256,
                 N_block=2,
                 num_quantiles=100,
                 beta_init=1.0,
                 pessimism=0.0,
                 std_mult=0.75,
                 learning_rate=3.0e-4,
                 gamma=0.99,
                 alpha_init=1.0,
                 tau_init=0.25,
                 target_entropy=0.5,
                 KL_target=0.05,
                 polyak=0.995,
                 start_steps=256,
                 buffer_length=1e6,
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
        # ALPHA is the regularization term/temperature for the entropy,
        # and is a learnable parameter!
        # BETA is the optimism of the policy, and is learnable as well!
        # TAU is the weight on the KL divergence
        # LR is the learning rate of the ``AdamW`` optimizers
        # POLYAK is the polyak averaging term
        # PESSIMISM is a fixed parameter, takes away from BETA steps
        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.LOGALPHA = nn.parameter.Parameter(torch.log(torch.tensor(alpha_init, device=device)))
        self.LOGBETA = nn.parameter.Parameter(torch.log(torch.tensor(beta_init, device=device)))
        self.LOGTAU = nn.parameter.Parameter(torch.log(torch.tensor(tau_init, device=device)))
        self.LR = learning_rate
        self.KL_TARGET = KL_target
        self.POLYAK = polyak
        self.PESSIMISM = pessimism
        
        # Get the number of state observations
        state, info = self.env.reset()
        self.state = state
        n_observations = len(state)
        self.scale = float(self.env.action_space.high[0])

        self.target_entropy = -target_entropy*float(self.env.action_space.shape[0])

        # Nets to predict the next action, keep it as the SAC!
        self.optimistic_net =BroNet(dim_in=n_observations, dim_out=1,dim_block=dim_block,dim_hid=dim_hid,N_block=N_block).to(device)
        self.pessimistic_net = BroNet(dim_in=n_observations, dim_out=2,dim_block=dim_block,dim_hid=dim_hid,N_block=N_block).to(device)
        self.std_mult = std_mult

        # Net to predict the reward for the next action;
        # BRO uses two like SAC, but averages them! 
        self.num_quantiles = num_quantiles
        self.bin_centers = ((0.5 + torch.arange(self.num_quantiles, device=device, dtype=torch.float32))/self.num_quantiles).unsqueeze(0).unsqueeze(-1)

        self.Q_net_1 = BroNet(dim_in=n_observations+1,dim_out=num_quantiles,dim_block=dim_block,dim_hid=dim_hid,N_block=N_block).to(device)
        self.Q_target_1 = BroNet(dim_in=n_observations+1,dim_out=num_quantiles,dim_block=dim_block,dim_hid=dim_hid,N_block=N_block).to(device)
        self.Q_target_1.load_state_dict(self.Q_net_1.state_dict())

        self.Q_net_2 = BroNet(dim_in=n_observations+1,dim_out=num_quantiles,dim_block=dim_block,dim_hid=dim_hid,N_block=N_block).to(device)
        self.Q_target_2 = BroNet(dim_in=n_observations+1,dim_out=num_quantiles,dim_block=dim_block,dim_hid=dim_hid,N_block=N_block).to(device)
        self.Q_target_2.load_state_dict(self.Q_net_2.state_dict())
        

        self.Q_optimizer = optim.AdamW([{"params":self.Q_net_1.parameters()},
                                       {"params":self.Q_net_2.parameters()}],
                                       lr=self.LR,weight_decay=1e-4)
        
        self.optimistic_optimizer = optim.AdamW(self.optimistic_net.parameters(), lr=self.LR, amsgrad=True, weight_decay=1e-4)
        self.pessimistic_optimizer = optim.AdamW(self.pessimistic_net.parameters(), lr=self.LR, amsgrad=True, weight_decay=1e-4)

        self.temperature_optimizer = optim.AdamW([self.LOGALPHA], lr=self.LR, amsgrad=True, weight_decay=1e-4)
        self.optimism_optimizer = optim.AdamW([self.LOGBETA], lr=0.1*self.LR, amsgrad=True, weight_decay=1e-4)
        self.tau_optimizer = optim.AdamW([self.LOGTAU], lr=0.1*self.LR, amsgrad=True, weight_decay=1e-4)
        
        self.memory = ReplayMemory(int(buffer_length))
    
        self.steps_done = 0
        self.start_steps = max(start_steps, 2*self.BATCH_SIZE)
        self.replay_ratio = replay_ratio
        
        self.episode_durations = []
        self.cumulative_rewards = []
        self.final_state_balanced = []
        self.final_state_truncated = []

    @property
    def temp(self):
        return self.LOGALPHA.exp()

    @property
    def optimism(self):
        return self.LOGBETA.exp()
    
    @property
    def KLweight(self):
        return self.LOGTAU.exp()

    def quantile_regression_loss(self, pred, target):
        assert pred.shape == target.shape, "Predictions and targets must align, Bxn_quantiles"
        target = target.unsqueeze(1)
        pred = pred.unsqueeze(-1)
        diff = target-pred #target_i - pred_j
        # Smooth element-wise Huber loss
        huber = torch.where(diff.abs() <= 1, 0.5 * diff.pow(2), diff.abs() - 0.5)
        # weight = bin center for positive difference, 1-bin center for negative
        weight = (self.bin_centers - (diff < 0).float()).abs().detach()
        return (weight * huber).sum(dim=1).mean()

    def calc_Qs(self, Q_in, target=False):
        if target:
            nets = [self.Q_target_1, self.Q_target_2]
        else:
            nets = [self.Q_net_1, self.Q_net_2]
        return [net(Q_in) for net in nets]

    def select_action(self, state, do_optimistic=False, do_deterministic=False):
        """
        We sample an action from the policy.

        We also want some exploration, so for the first `start_steps` steps, we instead uniformly sample
        actions from the environment
        """
        if (self.steps_done < self.start_steps) and (not do_deterministic):
            action = torch.as_tensor(self.env.action_space.sample(), device=device, dtype=torch.float32).view(1, 1)
        else:
            with torch.no_grad():
                action_mu, action_logsig = self.pessimistic_net(state).chunk(2, dim=-1)
                if do_optimistic:
                    shift = self.optimistic_net(state)
                    action_mu += shift
                    action_logsig += np.log(self.std_mult)
                if not do_deterministic:
                    action_logsig = action_logsig.clamp(-20, 2)
                    dist = torch.distributions.Normal(action_mu, torch.exp(action_logsig))
                    act = dist.sample()
                else:
                    act = action_mu
                action = self.scale*F.tanh(act)

        return torch.clamp(action, float(self.env.action_space.low[0]), float(self.env.action_space.high[0]))
        
    def optimize_model(self):
        # if our memory is shorter than the batch size, then 
        # we can't sample from it yet, so perform no changes
        if len(self.memory) < self.BATCH_SIZE:
            return
        # Now do everything in the loop (the train method will control the looping)
        # Step 1: sample batch
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
        if not non_final_mask.any():
            non_final_next_states = torch.tensor([])
        else:
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Step 2: Calculate critic target value (no clipped double Q)

        # Compute targets y(r, s', d) = r + gamma*(1-d)*min(Q_targ(s', a')) - alpha log pi(a', s')), a' ~ pi( s')
        # for each batch state according to target policy_net and Q_net
        next_state_values = torch.zeros((self.BATCH_SIZE, self.num_quantiles), device=device)
        if non_final_mask.any():
            with torch.no_grad():
                # we're not updating the targets yet, so don't accumulate grads
                next_action_mus, next_action_logsigmas = self.pessimistic_net(non_final_next_states).chunk(2, dim=-1)
                next_action_logsigmas = next_action_logsigmas.clamp(-20, 2)
                dists = torch.distributions.Normal(next_action_mus, torch.exp(next_action_logsigmas))
                u = dists.sample()
                next_action_values = self.scale*F.tanh(u)
                # see log prob def in appendix C here: https://arxiv.org/pdf/1801.01290
                logprobs = (dists.log_prob(u) - torch.log(1.0-torch.pow(F.tanh(u),2.0)+1e-12)-math.log(self.scale)).squeeze(-1)
                # non_final_next_states is shape N_batch_nonfinal x n_observations
                # next action values is shape N_batch_nonfinal x 1
                # concatenate along final axis
                target_input = torch.cat((non_final_next_states, next_action_values), dim=-1)
                # each Q net returns Bxnum_quantiles, so we want to average along the last dim
                Qs = self.calc_Qs(target_input, target=True)
                Qs = 0.5*(Qs[0] + Qs[1])
                next_state_values[non_final_mask] = self.GAMMA*(Qs-self.temp*logprobs.unsqueeze(-1))
        
        # Step 3: Update critic using pessimistic actor

        targets = reward_batch.unsqueeze(-1)+next_state_values
        Q_in = torch.cat((state_batch, action_batch), dim=-1)
        preds_1, preds_2 = self.calc_Qs(Q_in) #Bxnum_quantiles
        # Compute Huber loss around quantiles
        loss = self.quantile_regression_loss(preds_1, targets) + self.quantile_regression_loss(preds_2, targets)
        # Gradient descend the Q_net parameters
        self.Q_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q_net_1.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.Q_net_2.parameters(), 100)
        self.Q_optimizer.step()

        # Step 4: Calculate pessimistic actor value

        pes_action_mus, pes_action_logsigmas = self.pessimistic_net(state_batch).chunk(2, dim=-1)
        pes_action_logsigmas = pes_action_logsigmas.clamp(-20, 2)
        dists = torch.distributions.Normal(pes_action_mus, torch.exp(pes_action_logsigmas))
        u = dists.rsample()
        pes_action_values = self.scale*F.tanh(u)
        # again, logprob defined in appC here: https://arxiv.org/pdf/1801.01290
        logprobs_pes = (dists.log_prob(u) - torch.log(1.0-torch.pow(F.tanh(u),2.0)+1e-12)-math.log(self.scale)).squeeze(-1)
        Q_in_pes = torch.cat((state_batch, pes_action_values), dim=-1)
        # each Q net returns Bxnum_quantiles, so we want to average along last dim, then sum over last (remaining) dim
        Qs_pes = self.calc_Qs(Q_in_pes)
        Qs_pes = 0.5*(Qs_pes[0]+Qs_pes[1]) #Bxn_quantiles
        Qs_pes = torch.mean(Qs_pes, dim=-1) #Average over quantiles
        
        # Step 5: Update pessimistic actor

        # we want gradient _ascent_ so we use the negative of the sum of action values
        self.pessimistic_optimizer.zero_grad()
        # gradient ascend Q-entropy, so gradient descend -(Q-entropy)
        loss = -(Qs_pes-self.temp*logprobs_pes).sum()/self.BATCH_SIZE
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.pessimistic_net.parameters(), 100)
        self.pessimistic_optimizer.step()

        # Step 6: Calculate optimistic actor value

        opt_action_shift = self.optimistic_net(state_batch)
        opt_action_mus = pes_action_mus.detach() + opt_action_shift
        opt_action_logsigmas = pes_action_logsigmas.detach() + np.log(self.std_mult)
        opt_action_logsigmas = opt_action_logsigmas.clamp(-20, 2)
        dists = torch.distributions.Normal(opt_action_mus, torch.exp(opt_action_logsigmas))
        u = dists.rsample()
        opt_action_values = self.scale*F.tanh(u)
        # again, logprob defined in appC here: https://arxiv.org/pdf/1801.01290
        logprobs_opt = (dists.log_prob(u) - torch.log(1.0-torch.pow(F.tanh(u),2.0)+1e-12)-math.log(self.scale)).squeeze(-1)
        Q_in_opt = torch.cat((state_batch, opt_action_values), dim=-1)
        # each Q net returns Bxnum_quantiles, so we want to average along last dim, then sum over last (remaining) dim
        Qs_opt = self.calc_Qs(Q_in_opt)
        # average over Q nets, plus optimism times half absolute difference
        Qs_opt = (0.5*(Qs_opt[0]+Qs_opt[1]) + self.optimism *(Qs_opt[0]-Qs_opt[1]).abs()/2)
        # average over quantiles
        Qs_opt = torch.mean(Qs_opt, axis=-1) 

        # Step 7: Update optimistic actor

        # we want gradient _ascent_ so we use the negative of the sum of action values
        self.optimistic_optimizer.zero_grad()
        std_p = pes_action_logsigmas.detach().exp()
        mu_p = pes_action_mus.detach()
        std_o = opt_action_logsigmas.exp() / self.std_mult
        # pointwise K-L divergence for two Gaussians: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        KL =  (torch.log(std_p / std_o) + (std_o.pow(2) + (opt_action_mus -mu_p).pow(2)) / (2 * std_p.pow(2)) - 0.5)
        KL = KL.sum(-1) #sum over action dimension (only 1 for pendulum)
        loss = (-Qs_opt + self.KLweight*KL).sum()/self.BATCH_SIZE
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.optimistic_net.parameters(), 100)
        self.optimistic_optimizer.step()
        # we'll need the mean KL downstream
        delta = KL.detach().mean() / abs(self.env.action_space.shape[0]) - self.KL_TARGET

        # Step 8: Update entropy temperature

        self.temperature_optimizer.zero_grad()
        ent = -logprobs_pes.detach()
        loss = -self.temp*(self.target_entropy-ent).detach().sum()/self.BATCH_SIZE
        loss.backward()
        self.temperature_optimizer.step()

        # Step 9: Update optimism

        self.optimism_optimizer.zero_grad()
        term1 = self.optimism - self.PESSIMISM
        loss = term1*delta
        loss.backward()
        self.optimism_optimizer.step()

        # Step 10: Update KL weight

        self.tau_optimizer.zero_grad()
        loss = -self.KLweight * delta
        loss.backward()
        self.tau_optimizer.step()

        # Step 11: Update target networks
        # Polyak averaging to update target nets
        for net, target in zip([self.Q_net_1, self.Q_net_2], 
                                [self.Q_target_1, self.Q_target_2]):
            net_state_dict = net.state_dict()
            target_state_dict = target.state_dict()
            for key in net_state_dict:
                target_state_dict[key] = target_state_dict[key]*self.POLYAK + net_state_dict[key]*(1-self.POLYAK)
        
            target.load_state_dict(target_state_dict)

        torch.cuda.empty_cache()

    def run_sim(self, episode_number):
        frames = []
        state, info = self.log_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            # select action based on observed state
            action = self.select_action(state, do_optimistic=True, do_deterministic=True)
            # take step in environment to get next state, reward, and termination/truncation signals
            observation, reward, terminated, truncated, _ = self.log_env.step(action.item())
            # HxWx3
            frame = self.log_env.unwrapped.render(log_reward=True,reward=reward)
            frames.append(frame)
            # done signal is either terminated or truncated
            done = terminated or truncated
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        imageio.mimsave(vid_dir+f"/bro_ep{episode_number}.mp4", frames, fps=self.log_env.metadata["render_fps"])
            
    
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
                action = self.select_action(state, do_optimistic=True)
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
        
                    for i in range(self.replay_ratio):
                        self.optimize_model()
        
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
            plt.savefig(BRO_EPISODE_DURATION_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.cumulative_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Rewards')
            plt.savefig(BRO_CUM_REWARDS_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.final_state_truncated)
            plt.xlabel('Episode')
            plt.ylabel('Episode Truncated')
            plt.savefig(BRO_TRUNCATED_FIG, bbox_inches='tight')
            plt.clf()
            plt.plot(np.arange(num_episodes), self.final_state_balanced)
            plt.xlabel('Episode')
            plt.ylabel('Episode Success')
            plt.savefig(BRO_BALANCED_FIG, bbox_inches='tight')

    def dump(self):
        torch.save(self.optimistic_net.state_dict(), BRO_optimistic_pickle)
        torch.save(self.pessimistic_net.state_dict(), BRO_pessimistic_pickle)
        torch.save(self.Q_net_1.state_dict(), BRO_Q1_pickle)
        torch.save(self.Q_net_2.state_dict(), BRO_Q2_pickle)

    def load(self):
        self.optimistic_net.load_state_dict(torch.load(BRO_optimistic_pickle, weights_only=True, map_location=device))
        self.pessimistic_net.load_state_dict(torch.load(BRO_pessimistic_pickle, weights_only=True, map_location=device))
        self.Q_net_1.load_state_dict(torch.load(BRO_Q1_pickle, weights_only=True, map_location=device))
        self.Q_net_2.load_state_dict(torch.load(BRO_Q2_pickle, weights_only=True, map_location=device))
        self.Q_target_1.load_state_dict(self.Q_net_1.state_dict())
        self.Q_target_2.load_state_dict(self.Q_net_2.state_dict())
