import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from scipy.special import gamma

from src.utils import torch_onehot


def gumbel_softmax(log_prob_map, temperature, eps=1e-20):
    """Note that inputs are logprobs"""
    u = log_prob_map.new(log_prob_map.shape).uniform_(0, 1)
    g = -torch.log(-torch.log(u + eps) + eps)
    softmax_in = (log_prob_map + eps) + g
    y = F.softmax(softmax_in / temperature, dim=1)
    y_hard = torch.max(y, dim=1)[1]
    y_hard = torch_onehot(y_hard, y.shape[1]).type(y.type())
    return (y_hard - y).detach() + y


def gumbel_sigmoid(prob_map, temperature, eps=1e-20):
    U = prob_map.new(prob_map.shape).uniform_(0, 1)
    sigmoid_in = torch.log(prob_map + eps) - torch.log(1 - prob_map + eps) + torch.log(U + eps) - torch.log(1 - U + eps)
    y = torch.sigmoid(sigmoid_in / temperature)
    y_hard = torch.round(y)
    return (y_hard - y).detach() + y


class diag_w_Gauss_loglike(object):
    def __init__(self, μ=0, σ2=1, multiplier=1):
        super(diag_w_Gauss_loglike, self).__init__()
        self.μ = μ
        self.σ2 = σ2
        self.dist = Normal(self.μ, self.σ2)
        self.multiplier = multiplier

    def __call__(self, w):
        log_pW = self.dist.log_prob(w)
        return log_pW.sum() * self.multiplier

    def summary(self):
        return {"name": "diag_w_Gauss", "μ": self.μ, "σ2": self.σ2, "mul": self.multiplier}


class homo_Gauss_mloglike(nn.Module):
    def __init__(self, Ndims=1, sig=None):
        super(homo_Gauss_mloglike, self).__init__()
        if sig is None:
            self.log_std = nn.Parameter(torch.zeros(Ndims))
        else:
            self.log_std = nn.Parameter(torch.ones(Ndims) * np.log(sig), requires_grad=False)

    def forward(self, mu, y, model_std=None):
        sig = self.log_std.exp().clamp(min=1e-4)
        if model_std is not None:
            sig = (sig**2 + model_std**2)**0.5

        dist = Normal(mu, sig)
        return -dist.log_prob(y)


class pMOM_loglike(object):
    def __init__(self, r=1, τ=0.348, σ2=1.0, multiplier=1):
        super(pMOM_loglike).__init__()
        self.r = r
        self.τ = τ
        self.σ2 = σ2
        self.multiplier = multiplier

        self.d_p = -np.log(1 if self.r == 1 else 3 if self.r == 2 else 15)

    def __call__(self, W):
        p = W.numel()

        log_pW = self.d_p
        log_pW -= self.r * p * np.log(self.τ * self.σ2)
        log_pW += torch.sum(self.r * torch.log(W ** 2))
        log_pW += torch.distributions.normal.Normal(0, np.sqrt(self.τ * self.σ2)).log_prob(W).sum()

        return log_pW * self.multiplier

    def summary(self):
        return {"name": "pMOM", "r": self.r, "τ": self.τ, "σ2": self.σ2, "mul": self.multiplier}


class piMOM_loglike(object):
    def __init__(self, r=1, τ=0.348, σ2=1.0, multiplier=1):
        super(piMOM_loglike).__init__()
        self.r = r
        self.τ = τ
        self.σ2 = σ2
        self.multiplier = multiplier

        self.log_C = (-r + 1/2) * np.log(τ * σ2) + np.log(gamma(r - 1/2))

    def __call__(self, W):
        p = W.numel()

        log_pW = -p * self.log_C
        log_pW += torch.sum(-self.τ * self.σ2 / (W ** 2))
        log_pW += torch.sum(-self.r * torch.log(W ** 2))

        return log_pW * self.multiplier

    def summary(self):
        return {"name": "piMOM", "r": self.r, "τ": self.τ, "σ2": self.σ2, "mul": self.multiplier}


class peMOM_loglike(object):
    def __init__(self, r=None, τ=0.348, σ2=1.0, multiplier=1):
        super(peMOM_loglike).__init__()
        self.τ = τ
        self.σ2 = σ2
        self.multiplier = multiplier

        self.log_C = 0.5 * np.log(2 * np.pi * σ2 * τ) - (2/σ2)**0.5

    def __call__(self, W):
        p = W.numel()

        log_pW = -p * self.log_C
        log_pW += torch.sum(-W**2/(2 * self.σ2 * self.τ))
        log_pW += torch.sum(-self.τ/(W ** 2))

        return log_pW * self.multiplier

    def summary(self):
        return {"name": "pMOM", "τ": self.τ, "σ2": self.σ2, "mul": self.multiplier}

import torch
import torch.nn as nn
import torch.nn.functional as F


class depth_categorical(nn.Module):
    def __init__(self, prior_probs, prior_logprobs=None, cuda=True):
        # TODO: add option of specifying prior in terms of log_probs
        super(depth_categorical, self).__init__()

        self.prior_probs = torch.Tensor(prior_probs)
        assert self.prior_probs.sum().item() - 1 < 1e-6
        self.dims = self.prior_probs.shape[0]
        if prior_logprobs is None:
            self.logprior = self.prior_probs.log()
        else:
            self.logprior = torch.Tensor(prior_logprobs)
            self.prior_probs = self.logprior.exp()
            assert self.prior_probs.sum().item() - 1 < 1e-6

        self.current_posterior = None

        self.cuda = cuda
        if self.cuda:
            self.to_cuda()

    def to_cuda(self):
        self.prior_probs = self.prior_probs.cuda()
        self.logprior = self.logprior.cuda()

    @staticmethod
    def get_w_joint_loglike(prior_loglikes, act_vec, y, f_neg_loglike, N_train):
        """Note that if we average this to get exact joint, then all batches need to be the same size.
        Alternatively you can weigh each component with its batch size."""
        batch_size = act_vec.shape[1]
        depth = act_vec.shape[0]

        repeat_dims = [depth] + [1 for i in range(1, len(y.shape))]
        y_expand = y.repeat(*repeat_dims)  # targets are same across acts -> interleave
        act_vec_flat = act_vec.view(depth*batch_size, -1)  # flattening results in batch_n changing first
        loglike_per_act = -f_neg_loglike(act_vec_flat, y_expand).view(depth, batch_size)

        joint_loglike_per_depth = (N_train / batch_size) * loglike_per_act.sum(dim=1) + prior_loglikes  # (depth)
        return joint_loglike_per_depth

    def get_marg_loglike(self, joint_loglike_per_depth):
        log_joint_with_depth = joint_loglike_per_depth + self.logprior
        log_marginal_over_depth = torch.logsumexp(log_joint_with_depth, dim=0)
        return log_marginal_over_depth

    def get_depth_log_posterior(self, joint_loglike_per_depth, log_marginal_over_depth=None):
        if log_marginal_over_depth is None:
            log_marginal_over_depth = self.get_marg_loglike(joint_loglike_per_depth)
        log_joint_with_depth = joint_loglike_per_depth + self.logprior
        log_depth_posteriors = log_joint_with_depth - log_marginal_over_depth
        return log_depth_posteriors

    @staticmethod
    def marginalise_d_predict(act_vec, d_posterior, depth=None, softmax=False, get_std=False):
        """ Predict while marginalising d with given distribution."""
        # TODO: switch to logprobs and log q
        assert not (softmax and get_std)
        if softmax:
            preds = F.softmax(act_vec, dim=2)
        else:
            preds = act_vec

        q = d_posterior.clone().detach()
        while len(q.shape) < len(act_vec.shape):
            q = q.unsqueeze(1)

        if get_std:
            pred_mu = (q * preds).sum(dim=0)
            model_var = (q * preds**2).sum(dim=0) - pred_mu**2
            return pred_mu, model_var.pow(0.5)

        weighed_preds = q * preds
        return weighed_preds.sum(dim=0)


class depth_categorical_VI(depth_categorical):

    def __init__(self, prior_probs, cuda=True, eps=1e-35):
        super(depth_categorical_VI, self).__init__(prior_probs, None, cuda)

        self.q_logits = nn.Parameter(torch.zeros(self.dims), requires_grad=True)
        self.eps = eps
        if cuda:
            self.to_cuda_VI()

    def to_cuda_VI(self):
        self.q_logits.data = self.q_logits.data.cuda()

    def get_q_logprobs(self):
        """Get logprobs of each depth configuration"""
        return F.log_softmax(self.q_logits, dim=0)

    def get_q_probs(self):
        """Get probs of each depth configuration"""
        return F.softmax(self.q_logits, dim=0)

    def get_KL(self):
        """KL between categorical distributions"""
        log_q = self.get_q_logprobs()
        q = self.get_q_probs().clamp(min=self.eps, max=(1 - self.eps))
        log_p = self.logprior
        KL = (q * (log_q - log_p)).sum(dim=0)
        return KL

    def get_E_loglike(self, joint_loglike_per_depth):
        """Calculate ELBO with deterministic expectation."""
        q = self.get_q_probs()
        E_loglike = (q * joint_loglike_per_depth).sum(dim=0)
        return E_loglike

    def estimate_ELBO(self, prior_loglikes, act_vec, y, f_neg_loglike, N_train, Beta=1):
        """Estimate ELBO on logjoint of data and network weights"""
        joint_loglike_per_depth = depth_categorical.get_w_joint_loglike(prior_loglikes, act_vec, y,
                                                                        f_neg_loglike, N_train)
        # print('Elbo joint loglike per depth', joint_loglike_per_depth)
        E_loglike = self.get_E_loglike(joint_loglike_per_depth)
        KL = self.get_KL()
        return E_loglike - Beta * KL

    def q_predict(self, act_vec, depth=None, softmax=False):
        """Predict marginalising depth with approximate posterior. Currently this will only support classification"""
        return depth_categorical.marginalise_d_predict(act_vec, self.get_q_probs(), depth=depth, softmax=softmax)
