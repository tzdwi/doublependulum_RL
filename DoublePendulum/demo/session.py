from typing import Literal
import numpy as np
import torch

from .policies import MODEL_IDS, POLICIES, load
from DoublePendulum.nn.nn_common import init_env, device

class Session:
	"""
	Our object for the demo that holds all the public-facing
	and necessary info to run the demo.
	"""
	def __init__(self, model_id : Literal[*MODEL_IDS]):
		# Initialize the environment...
		self.env = init_env(render_mode="rgb_array")
		# and the model...
		self.policy = load(model_id)
		# and the action selector...
		if self.policy is None:
			print("Error in loading policy model")
			self.action = lambda *args: torch.zeros(1,1)
		else:
			self.action = POLICIES[model_id]["action"]

		self.playing = None
	
	def start(self):
		# reset environment
		self.env.reset(options={"balanced":True})
		# set playing flag to False
		self.playing = False
		# and return a frame
		return self.env.unwrapped.render()

	def pick_model(self, model_id : Literal[*MODEL_IDS]):
		self.policy = load(model_id)
		# Reinitialize with a different model
		if self.policy is None:
			print("Error in loading policy model")
			self.action = lambda *args: torch.zeros(1,1)
		else:
			self.action = POLICIES[model_id]["action"]
		return self.start()

	def kick(self, scale=0.1):
		# a small kick, scaled by the user-set slider bar
		self.env.unwrapped._agent_location = scale*np.random.randn()
		
		# record an observation
		obs = self.env.unwrapped._get_obs()
		self.env.unwrapped.state = obs

		# we're playing now!
		self.playing = True

		return self.env.unwrapped.render()

	def step(self):
		if not self.playing:
			return self.env.unwrapped.render(), "holding"
		else:
			obs = self.env.unwrapped._get_obs()
			obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
			with torch.no_grad():
				action = self.action(obs, self.policy)
			observation, reward, terminated, truncated, _ = self.env.step(action.item())
			if terminated or truncated:
				self.playing = False
			if reward == 1000:
				status = "balanced"
			elif reward == -100:
				status = "off_track"
			elif truncated:
				status = "timeout"
			else:
				status = "working"
			return self.env.unwrapped.render(), status

