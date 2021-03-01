import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
import a2c_ppo_acktr.networks as nets


class Agent(nn.Module):
    def __init__(self, actor, critic):
        super(Agent, self).__init__()
        self.actor = actor
        self.critic = critic

    @property
    def is_recurrent(self):
        return self.actor.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.actor.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        action, action_log_probs, rnn_hxs = self.actor.act(inputs, rnn_hxs, masks)
        value = self.get_value(inputs, rnn_hxs, masks)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value = self.critic(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value = self.get_value(inputs, rnn_hxs, masks)
        dist = self.actor.get_dist(inputs, rnn_hxs, masks)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs