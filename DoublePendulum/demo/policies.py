import torch
import torch.nn.functional as F
from pathlib import Path

from DoublePendulum.nn.sac import SAC_POLICY_DP, SAC_policy_pickle
from DoublePendulum.nn.ddpg import DDPG_POLICY_DP, DDPG_policy_pickle

from DoublePendulum.nn.nn_common import device

def sac_policy_builder(n_observations):
	if not Path(SAC_policy_pickle).is_file():
		return None
	net = SAC_POLICY_DP(n_observations).to(device)
	try:
		net.load_state_dict(torch.load(SAC_policy_pickle, weights_only=True, map_location=device))
		net.eval()
	except RuntimeError:
		print("Failed assigning pre-saved weights, returning empty model")
		net = None
	except OSError:
		print("Failed loading pre-saved weights, returning empty model")
		net = None
	except Exception as e:
		print("Something unexpected happened: ")
		print(e)
		net = None
	return net

def ddpg_policy_builder(n_observations):
	if not Path(DDPG_policy_pickle).is_file():
		return None
	net = DDPG_POLICY_DP(n_observations, scale=10.0).to(device)
	try:
		net.load_state_dict(torch.load(DDPG_policy_pickle, weights_only=True, map_location=device))
		net.eval()
	except RuntimeError:
		print("Failed assigning pre-saved weights, returning empty model")
		net = None
	except OSError:
		print("Failed loading pre-saved weights, returning empty model")
		net = None
	except Exception as e:
		print("Something unexpected happened: ")
		if hasattr(e, 'message'):
			print(e.message)
		else:
			print(e)
		net = None
	return net

def sac_action(state, SAC_POLICY_NET):
	action_mu, _ = SAC_POLICY_NET(state).chunk(2, dim=-1)
	action = 10.0*F.tanh(action_mu)
	return action

def ddpg_action(state, DDPG_POLICY_NET):
	action = DDPG_POLICY_NET(state)
	return action

# Entries are model_name : (label, function that loads in the net, action function)
POLICIES = {
	'sac':{
		"Label":"Soft Actor-Critic",
		"load":sac_policy_builder,
		"action":sac_action
		},
	'ddpg':{
		"Label":"Deep Deterministic Policy Gradient",
		"load":ddpg_policy_builder,
		"action":ddpg_action
		},
	}

MODEL_IDS = tuple(POLICIES.keys())

def load(model_id):
	reg = POLICIES[model_id]
	net = reg["load"](7)
	return net
