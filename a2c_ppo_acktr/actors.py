import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedNormal
from a2c_ppo_acktr.utils import init
import a2c_ppo_acktr.networks as nets


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
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

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_dist(self, inputs, rnn_hxs, masks):
        actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        # TODO:rnn_hxs are not used!
        return dist

    def get_dist_params(self, inputs, rnn_hxs, masks):
        actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        params = self.dist.get_params(actor_features)
        # TODO:rnn_hxs are not used!
        return params

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        dist = self.get_dist(inputs, rnn_hxs, masks)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return action, action_log_probs, rnn_hxs


class Gating(Policy):
    def __init__(self, obs_shape, num_primitives, base=None, base_kwargs=None):
        nn.Module.__init__(self)
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
        self.dist = DiagGaussian(self.base.output_size, num_primitives)

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        dist = self.get_dist(inputs, rnn_hxs, masks)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()

        action = F.sigmoid(action)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return action, action_log_probs, rnn_hxs


class Composite(Policy):
    def __init__(self, primitives, gating):
        nn.Module.__init__(self)
        self.primitives = primitives
        self.gating = gating

    @property
    def is_recurrent(self):
        return self.gating.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.gating.recurrent_hidden_state_size

    def execute_primitives(self, inputs, rnn_hxs, masks):
        mus, logstds = zip(*[p.get_dist_params(inputs, rnn_hxs, masks) for p in self.primitives])  # list of length k of (bsize, adim)
        mus = torch.stack(mus, dim=1)  # (bsize, k, outdim)
        stds = torch.exp(torch.stack(logstds, dim=1))  # (bsize, k, outdim)
        return mus, stds

    def compose_mu(self, mus, weights_over_variance, inverse_variance):
        weighted_mus = weights_over_variance * mus
        composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
        return composite_mu

    def compose_params(self, mus, stds, weights):
        weights_over_variance = weights/(stds*stds)  # (bsize, k, zdim)
        inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
        ##############################
        # composite_std = 1.0/torch.sqrt(inverse_variance)
        composite_logstd = -0.5 * torch.log(inverse_variance)
        ##############################
        composite_mu = self.compose_mu(mus, weights_over_variance, inverse_variance)  # (bsize, zdim)
        return composite_mu, composite_logstd

    def get_dist_params(self, inputs, rnn_hxs, masks):
        weights, _, _ = self.gating.act(inputs, rnn_hxs, masks)
        weights = weights.unsqueeze(-1)  #  (bsize, K, 1)
        # TODO:rnn_hxs are not used!
        mus, stds = self.execute_primitives(inputs, rnn_hxs, masks)
        composite_mu, composite_logstd = self.compose_params(mus, stds, weights)
        return composite_mu, composite_logstd

    def get_dist(self, inputs, rnn_hxs, masks):
        mu, logstd = self.get_dist_params(inputs, rnn_hxs, masks)
        dist = FixedNormal(mu, logstd.exp())
        return dist

