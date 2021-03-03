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

        deterministic = True  # hacky for now

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

    def num_primitives(self):
        return len(self.primitives)

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

    def get_gating(self, inputs, rnn_hxs, masks):
        weights, _, _ = self.gating.act(inputs, rnn_hxs, masks)
        weights = torch.sigmoid(weights)  # FOR NOW
        weights = weights.unsqueeze(-1)  #  (bsize, K, 1)
        return weights

    def get_dist_params(self, inputs, rnn_hxs, masks):
        weights = self.get_gating(inputs, rnn_hxs, masks)
        # TODO:rnn_hxs are not used!
        mus, stds = self.execute_primitives(inputs, rnn_hxs, masks)
        composite_mu, composite_logstd = self.compose_params(mus, stds, weights)
        return composite_mu, composite_logstd

    def get_dist(self, inputs, rnn_hxs, masks):
        mu, logstd = self.get_dist_params(inputs, rnn_hxs, masks)
        dist = FixedNormal(mu, logstd.exp())
        return dist

class EfficientComposite(Composite):
    def __init__(self, primitives, gating):
        Composite.__init__(self, primitives, gating)
        self.higher_level_gatings = nn.ModuleList([])

    def add_gating(self, gating):
        self.higher_level_gatings.append(gating)

    def num_primitives(self):
        return len(self.primitives) + len(self.higher_level_gatings)

    def get_gating(self, inputs, rnn_hxs, masks):
        base_term, _, _ = self.gating.act(inputs, rnn_hxs, masks)  # (bsize, K)
        base_term = torch.sigmoid(base_term)  # temporary
        K = base_term.shape[1]

        # everything in weights_trace has shape (bsize, K)
        weights_trace = [base_term]
        for gating in self.higher_level_gatings:
            higher_weights, _, _ = gating.act(inputs, rnn_hxs, masks)
            higher_weights = torch.sigmoid(higher_weights)  # temporary

            assert len(weights_trace) == higher_weights.shape[1] - K

            new_term = higher_weights[:, :K]  # (bsize, K)
            for j in range(len(weights_trace)):
                weights_trace[j] = torch.einsum('bk,b->bk', weights_trace[j], higher_weights[:, K+j])
                new_term = new_term + weights_trace[j]
            weights_trace.append(new_term)

        weights = weights_trace[-1]

        # weights = torch.sigmoid(weights)  
        weights = weights.unsqueeze(-1)  #  (bsize, K, 1)

        return weights

