import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
import a2c_ppo_acktr.networks as nets


class Critic(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Critic, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = nets.CNNBase
                base_kwargs['hidden_size'] = 512
            elif len(obs_shape) == 1:
                base = nets.MLPBase
                base_kwargs['hidden_size'] = 64
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linear = init_(nn.Linear(base_kwargs['hidden_size'], 1))


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        value, _ = self.base(inputs, rnn_hxs, masks)
        value = self.critic_linear(value)
        return value