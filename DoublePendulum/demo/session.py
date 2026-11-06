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

	def kick(self, scale=5.0):
		# Slider value is a force in newtons. The sign is random, and the force is held for one decision.
		plant = self.env.unwrapped
		magnitude = float(np.clip(scale, 0.0, plant.max_F))
		sign = -1.0 if np.random.random() < 0.5 else 1.0
		plant._shove(sign * magnitude)
		self.playing = True
		return plant.render()

	def step(self):
		if not self.playing:
			return self.env.unwrapped.render(), "holding"
		else:
			obs = self.env.unwrapped._get_obs()
			obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
			with torch.no_grad():
				action = self.action(obs, self.policy)
			observation, reward, terminated, truncated, info = self.env.step(action.item())
			if terminated or truncated:
				self.playing = False
			if info["success"]:
				status = "balanced"
			elif info["offtrack"]:
				status = "off_track"
			elif truncated:
				status = "timeout"
			else:
				status = "working"
			return self.env.unwrapped.render(), status

