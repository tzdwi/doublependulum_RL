"""
We're going to be training a Soft Actor-Critic with an GRN component
https://arxiv.org/pdf/2509.25756
"""
import math
import os
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

from .nn_common import this_dir, plot_dir, init_env, device, Transition, ReplayMemory, warmup_action, EarlyStop, polyak_update, prepare_parallel, run_parallel, LQR_OBS_CENTER, LQR_OBS_SPREAD, _maybe_record, _record_final, _shift_training_seed
from .sac import SAC_Q_DP

SAC_G_velocity_pickle = this_dir+"/pickles/sac_flowg_velocity_net.pt"
SAC_G_Q1_pickle = this_dir+"/pickles/sac_flowg_q1_net.pt"
SAC_G_Q2_pickle = this_dir+"/pickles/sac_flowg_q2_net.pt"
SAC_G_velocity_best_pickle = this_dir+"/pickles/sac_flowg_velocity_net_best.pt"
SAC_G_Q1_best_pickle = this_dir+"/pickles/sac_flowg_q1_net_best.pt"
SAC_G_Q2_best_pickle = this_dir+"/pickles/sac_flowg_q2_net_best.pt"

SAC_G_EPISODE_DURATION_FIG = plot_dir+"/sac_flowg_episode_duration.png"
SAC_G_CUM_REWARDS_FIG = plot_dir+"/sac_flowg_cum_rewards.png"
SAC_G_BALANCED_FIG = plot_dir+"/sac_flowg_balanced.png"
SAC_G_TRUNCATED_FIG = plot_dir+"/sac_flowg_truncated.png"

def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.

	Borrowed from https://discuss.pytorch.org/t/sinusoidal-encoding-and-gradient-flow/200472

	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""
	half = dim // 2
	freqs = torch.exp(
		-math.log(max_period)
		* torch.arange(half, device=timesteps.device, dtype=torch.float32)
		/ max(half - 1, 1)
	)
	args = timesteps.float().reshape(-1, 1) * freqs.unsqueeze(0)
	embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
	if dim % 2:
		embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
	return embedding

class TIME_EMB_MLP(nn.Module):
	"""
	Returns a sinusoidal embedding run through an MLP
	"""
	def __init__(self, layers):
		super().__init__()
		self.sin_dim = layers[0]
		self.layers = nn.ModuleList(
			nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
		)
		self.activation = nn.SiLU()
	def forward(self, time):
		x = timestep_embedding(time, self.sin_dim)
		for i, layer in enumerate(self.layers):
			x = layer(x)
			if i + 1 < len(self.layers):
				x = self.activation(x)
		return x

class SAC_FLOWG_VELOCITY(nn.Module):
	"""
	A Gated Recurrent Net Unit that will be used to derive velocities
	to transform proposed actions along the net
	"""
	def __init__(self, n_observations, n_actions, t_emb_dim=32, t_emb_hid=64, d_hid=128, vscale=50):
		super(SAC_FLOWG_VELOCITY, self).__init__()
		self.gate_layer1 = nn.Linear(n_observations+n_actions+t_emb_dim, d_hid)
		self.gate_layer2 = nn.Linear(d_hid, n_actions)
		nn.init.zeros_(self.gate_layer2.weight)
		self.gate_layer2.bias.data[:] = 1.0

		self.layer1 = nn.Linear(n_observations+n_actions+t_emb_dim, d_hid)
		self.layer2 = nn.Linear(d_hid, n_actions)

		self.std_layer1 = nn.Linear(n_observations, d_hid)
		self.std_layer2 = nn.Linear(d_hid, n_actions)
		nn.init.zeros_(self.std_layer2.weight)
		# -1.5 + 3.5*tanh(bias) = log(0.1).
		self.std_layer2.bias.data[:] = math.atanh((math.log(0.1) + 1.5) / 3.5)
		
		self.activation = nn.SiLU()
		self.sigmoid = nn.Sigmoid()

		self.tanh = nn.Tanh()
		self.vscale = vscale

		self.time_emb = TIME_EMB_MLP(layers=(t_emb_dim, t_emb_hid, t_emb_dim))

	def forward(self, state, action, time):
		time = self.time_emb(time)
		x = torch.cat((state, action, time), dim=-1) # concatenate along non-batch dim
		vhat = self.vscale*self.tanh(self.layer2(self.activation(self.layer1(x))))
		gate = self.gate_layer2(self.activation(self.gate_layer1(x)))
		log_std = self.tanh(self.std_layer2(self.activation(self.std_layer1(state))))
		log_std = -5 + 3.5 * (log_std + 1)
		return self.sigmoid(gate) * (vhat - action), log_std

	def sample(self, state, K, scale, do_deterministic=False):
		if do_deterministic:
			A = torch.zeros(state.shape[0], 1, device=state.device)
		else:
			A = torch.randn(state.shape[0], 1, device=state.device)
		logp = torch.distributions.Normal(0, 1).log_prob(A).sum(-1)
		dt = 1.0 / K
		time_tens = torch.ones(state.shape[0], dtype=torch.float32, device=device)
		for i in range(K):
			time = i*time_tens/K
			v, log_std = self.forward(state, A, time)
			if do_deterministic:
				A = A + v * dt
				continue
			# Noise-augmented rollout, Appendix A.1: drift-correct v so the
			# injected per-step noise leaves the deterministic path's final
			# action marginal unchanged (b_theta(0,.,.) = v_theta(0,.,.)).
			t = time.unsqueeze(-1)
			remaining = (1.0 - t).clamp(min=1e-3)
			log_std = log_std.clamp(-20, 2)
			sigma = log_std.exp()
			drift = (1.0 + t * sigma.pow(2) / (2.0 * remaining)) * v - (sigma.pow(2) / (2.0 * remaining)) * A
			# Appendix A separately defines the t=0 value, where the displayed
			# t/t contraction term is undefined, as b_theta(0,.,.) = v_theta.
			drift = torch.where(t == 0.0, v, drift)
			mean = A + drift * dt
			std = sigma * math.sqrt(dt)
			A = mean + std * torch.randn_like(mean)
			logp = logp + torch.distributions.Normal(mean, std).log_prob(A).sum(-1)
		y = torch.tanh(A)
		logp = logp - torch.log(scale * (1 - y**2) + 1e-6).sum(-1)
		action = scale * y
		return action, logp


class SAC_FLOWG_Learner:
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
				 K=4,
				 gamma=0.99,
				 tau=0.995,
				 alpha=0.03,
				 learning_rate=3e-4,
				 policy_delay=2,
				 actor_holdback=20000,
				 start_steps=2000,
				 buffer_length=1e6,
				 log_progress=False,
				 obs_center=LQR_OBS_CENTER,
				 obs_spread=LQR_OBS_SPREAD,
				 reward_scale=1.0,
				 freeze_alpha=False,
				 adam_betas=(0.9, 0.999),
				 num_envs=1):

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
		self.video_prefix = "sac_flowg"
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
		# TAU is the update rate of the target networks for polyak averaging
		# alpha is the initial value of the learned entropy temperature
		# LR is the learning rate of the ``AdamW`` optimizer
		
		self.BATCH_SIZE = batch_size
		self.policy_delay = policy_delay
		self.actor_holdback = actor_holdback
		self.GAMMA = gamma
		self.TAU = tau
		self.LR = learning_rate
		
		# Get the number of state observations
		state, info = self.env.reset()
		self.state = state
		n_observations = len(state)
		n_actions = self.env.action_space.shape[0]
		self.scale = float(self.env.action_space.high[0])
		self.reward_scale = reward_scale
		self.obs_center = None if obs_center is None else torch.as_tensor(obs_center, dtype=torch.float32, device=device)
		self.obs_spread = None if obs_spread is None else torch.as_tensor(obs_spread, dtype=torch.float32, device=device)
		self.freeze_alpha = freeze_alpha

		self.log_alpha = torch.tensor(math.log(alpha), device=device, requires_grad=True)
		self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.LR)
		# log probs are already scaled properly so this should just be 0
		self.target_entropy = 0.0 

		# Net to predict velocities to adapt proposed actions
		self.velocity_net = SAC_FLOWG_VELOCITY(n_observations, n_actions).to(device)
		
		# Nets to predict the reward for the next action, which we take the minimum of
		# otherwise the policy just learns any faults in the one Q net
		self.Q_net_1 = SAC_Q_DP(n_observations).to(device)
		self.Q_net_2 = SAC_Q_DP(n_observations).to(device)
		# target nets initialized with same weights
		self.Q_target_1 = SAC_Q_DP(n_observations).to(device)
		self.Q_target_1.load_state_dict(self.Q_net_1.state_dict())
		self.Q_target_2 = SAC_Q_DP(n_observations).to(device)
		self.Q_target_2.load_state_dict(self.Q_net_2.state_dict())
		
		self.Q_optimizer = optim.AdamW([{"params":self.Q_net_1.parameters()},
									   {"params":self.Q_net_2.parameters()}],
									   lr=self.LR, betas=adam_betas, amsgrad=True)
		self.velocity_optimizer = optim.AdamW(self.velocity_net.parameters(), 
											lr=self.LR, betas=adam_betas, amsgrad=True)
		# Mean squared error loss
		self.criterion = nn.MSELoss()
		self.memory = ReplayMemory(int(buffer_length))

		# number of steps for the recurrent layer
		self.K = K
	
		self.steps_done = 0
		self.start_steps = max(start_steps, 2*self.BATCH_SIZE)
		
		self.episode_durations = []
		self.cumulative_rewards = []
		self.final_state_balanced = []
		self.final_state_truncated = []
		prepare_parallel(self, num_envs,
						 size=size, max_F=max_F, max_ang_vel=max_ang_vel,
						 mm=mm, m1=m1, l1=l1, m2=m2, l2=l2, dt=dt, theta_tol=theta_tol)

	def select_action(self, state, do_deterministic=False):
		"""
		We sample an action from the policy.

		We also want some exploration, so for the first `start_steps` steps, we instead uniformly sample
		actions from the environment
		"""
		if (self.steps_done < self.start_steps) and (not do_deterministic):
			action = torch.as_tensor(warmup_action(self.env.action_space), device=device, dtype=torch.float32).view(1, 1)
			logp = None
		else:
			with torch.no_grad():
				action, logp = self.velocity_net.sample(self._condition(state),self.K, self.scale, do_deterministic=do_deterministic)

		return torch.clamp(action, float(self.env.action_space.low[0]), float(self.env.action_space.high[0])), logp

	def _condition(self, state):
		if self.obs_center is None:
			return state
		return (state - self.obs_center) / self.obs_spread
		
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
	
		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											  batch.next_state)), device=device, dtype=torch.bool)
		if not non_final_mask.any():
			non_final_next_states = torch.tensor([])
		else:
			non_final_next_states = self._condition(torch.cat([s for s in batch.next_state
													if s is not None]))
		state_batch = self._condition(torch.cat(batch.state))
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward) / self.reward_scale

		# Compute Q(s, a)
		# state_batch is shape N_batch x n_observations
		# action_batch is shape N_batch x 1
		# concatenate along final axis
		Q_in = torch.cat((state_batch, action_batch / self.scale), dim=-1)
		state_action_values_1 = self.Q_net_1(Q_in)
		state_action_values_2 = self.Q_net_2(Q_in)
	
		# Compute targets y(r, s', d) = r + gamma*(1-d)*min(Q_targ(s', a')) - alpha log pi(a', s')), a' ~ pi( s')
		# for each batch state according to target policy_net and Q_net
		next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
		if non_final_mask.any():
			with torch.no_grad():
				# we're not updating the targets yet, so don't accumulate grads
				next_action_values, logprobs = self.velocity_net.sample(non_final_next_states, self.K, self.scale)
				# non_final_next_states is shape N_batch_nonfinal x n_observations
				# next action values is shape N_batch_nonfinal x 1
				# concatenate along final axis
				target_input = torch.cat((non_final_next_states, next_action_values / self.scale), dim=-1)
				# each Q net returns Bx1, so we want to stack along the last dim
				Qs = torch.cat([Q_target(target_input) for Q_target in [self.Q_target_1, self.Q_target_2]], dim=-1)
				selected = torch.min(Qs, dim=-1).values
				
				next_state_values[non_final_mask] = self.GAMMA*(selected-self.log_alpha.exp()*logprobs)

		targets = reward_batch+next_state_values
	
		# Compute MSE loss
		loss = self.criterion(state_action_values_1.squeeze(-1), targets) + self.criterion(state_action_values_2.squeeze(-1), targets)
	
		# Gradient descend the Q_net parameters
		self.Q_optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.Q_net_1.parameters(), 100)
		torch.nn.utils.clip_grad_value_(self.Q_net_2.parameters(), 100)
		self.Q_optimizer.step()

		if not do_policy_step:
			return
		# Now gradient ascend the velocity_net parameters
		next_action_values, logprobs = self.velocity_net.sample(state_batch, self.K, self.scale)
		Q_in = torch.cat((state_batch, next_action_values / self.scale), dim=-1)
		# each Q net returns Bx1, so we want to stack along the last dim
		Qs = torch.cat([Q_net(Q_in) for Q_net in [self.Q_net_1, self.Q_net_2]], dim=-1)
		selected = torch.min(Qs, dim=-1).values
		# we want gradient _ascent_ so we use the negative of the sum of action values
		loss = -(selected-self.log_alpha.exp().detach()*logprobs).sum()/self.BATCH_SIZE
		self.velocity_optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.velocity_net.parameters(), 100)
		self.velocity_optimizer.step()

		if self.freeze_alpha:
			return
		alpha_loss = -(self.log_alpha * (logprobs.detach() + self.target_entropy)).mean()
		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

	def _actor_started(self):
		return self.steps_done > self.start_steps + self.actor_holdback

	def _parallel_update(self, step_index, early):
		if len(self.memory) > self.start_steps:
			actor_ready = self._actor_started()
			self.optimize_model(do_policy_step=actor_ready and (step_index % self.policy_delay == 0))
			for net, target in zip([self.Q_net_1, self.Q_net_2],
								   [self.Q_target_1, self.Q_target_2]):
				polyak_update(net, target, self.TAU)

	def train(self, progress=False, make_plots=False, num_episodes = None, start_episode=0):
		if not num_episodes:
			if torch.cuda.is_available() or torch.backends.mps.is_available():
				num_episodes = 600
			else:
				num_episodes = 50

		if self.num_envs > 1:
			run_parallel(
				self, num_episodes, start_episode,
				lambda states: [self.select_action(state)[0] for state in states],
				self._parallel_update, self._actor_started, progress)
			return

		iterator = range(start_episode, num_episodes)
		if progress:
			iterator = tqdm(iterator)

		early = EarlyStop()
		stop = None
		for i_episode in iterator:
			_maybe_record(self, i_episode)
			cumulative_reward = 0
			# Initialize the environment and get its state
			state, info = self.env.reset(seed=_shift_training_seed(i_episode))
			state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
			for t in count():
				# select action based on observed state
				action, _ = self.select_action(state)
				# take step in environment to get next state, reward, and termination/truncation signals
				observation, reward, terminated, truncated, info = self.env.step(action.item())
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
					actor_ready = self.steps_done > self.start_steps + self.actor_holdback
					self.optimize_model(do_policy_step=actor_ready and (t % self.policy_delay == 0))
			
					# Polyak averaging to update target nets
					for net, target in zip([self.Q_net_1, self.Q_net_2],
										   [self.Q_target_1, self.Q_target_2]):
						polyak_update(net, target, self.TAU)
		
				if done:
					end_episode = getattr(self.memory, "end_episode", None)
					if end_episode is not None:
						end_episode(bool(info["success"]))
					self.episode_durations.append(t + 1)
					self.cumulative_rewards.append(cumulative_reward)
					if info["success"]:
						self.final_state_balanced.append(1)
					else:
						self.final_state_balanced.append(0)
					if truncated:
						self.final_state_truncated.append(1)
					else:
						self.final_state_truncated.append(0)
					stop = early.check(
						self.episode_durations, self.final_state_balanced,
						self.steps_done > self.start_steps + self.actor_holdback)
					if early.improved or (stop or "").startswith("solved"):
						self.dump_best(len(self.episode_durations), early.best)
					break
			if stop:
				print(stop)
				break
		_record_final(self)
		if make_plots:
			fig = plt.figure(dpi=300)
			plt.plot(np.arange(len(self.episode_durations)), self.episode_durations)
			plt.xlabel('Episode')
			plt.ylabel('Episode Duration')
			plt.savefig(SAC_G_EPISODE_DURATION_FIG, bbox_inches='tight')
			plt.clf()
			plt.plot(np.arange(len(self.episode_durations)), self.cumulative_rewards)
			plt.xlabel('Episode')
			plt.ylabel('Cumulative Rewards')
			plt.savefig(SAC_G_CUM_REWARDS_FIG, bbox_inches='tight')
			plt.clf()
			plt.plot(np.arange(len(self.episode_durations)), self.final_state_truncated)
			plt.xlabel('Episode')
			plt.ylabel('Episode Truncated')
			plt.savefig(SAC_G_TRUNCATED_FIG, bbox_inches='tight')
			plt.clf()
			plt.plot(np.arange(len(self.episode_durations)), self.final_state_balanced)
			plt.xlabel('Episode')
			plt.ylabel('Episode Success')
			plt.savefig(SAC_G_BALANCED_FIG, bbox_inches='tight')

	def dump(self):
		os.makedirs(os.path.dirname(SAC_G_velocity_pickle), exist_ok=True)
		torch.save(self.velocity_net.state_dict(), SAC_G_velocity_pickle)
		torch.save(self.Q_target_1.state_dict(), SAC_G_Q1_pickle)
		torch.save(self.Q_target_2.state_dict(), SAC_G_Q2_pickle)

	def dump_best(self, episode, mean_length):
		os.makedirs(os.path.dirname(SAC_G_velocity_best_pickle), exist_ok=True)
		torch.save(self.velocity_net.state_dict(), SAC_G_velocity_best_pickle)
		torch.save(self.Q_target_1.state_dict(), SAC_G_Q1_best_pickle)
		torch.save(self.Q_target_2.state_dict(), SAC_G_Q2_best_pickle)
		length = "n/a" if mean_length is None else f"{mean_length:.0f}"
		print(f"saved best policy at episode {episode}, mean length {length}")

	def load(self, best=True):
		velocity_path = SAC_G_velocity_best_pickle if best else SAC_G_velocity_pickle
		q1_path = SAC_G_Q1_best_pickle if best else SAC_G_Q1_pickle
		q2_path = SAC_G_Q2_best_pickle if best else SAC_G_Q2_pickle
		self.velocity_net.load_state_dict(torch.load(velocity_path, weights_only=True, map_location=device))
		self.Q_target_1.load_state_dict(torch.load(q1_path, weights_only=True, map_location=device))
		self.Q_target_2.load_state_dict(torch.load(q2_path, weights_only=True, map_location=device))

		self.Q_net_1.load_state_dict(self.Q_target_1.state_dict())
		self.Q_net_2.load_state_dict(self.Q_target_2.state_dict())
