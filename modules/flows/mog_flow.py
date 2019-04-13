import torch
import torch.nn as nn
from modules.flows.flow import Flow
from overrides import overrides
import numpy as np
import math
from tools.utils import init_linear_layer
from scipy.stats import ortho_group
from scipy.linalg import lu
import torch.nn.functional as F

def logsumexp(x, dim=None):
    """
    Args:
    x: A pytorch tensor (any dimension will do)
    dim: int or None, over which to perform the summation. `None`, the
    default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))


class MogFlow_batch(Flow):
    def __init__(self, args, var):
        # freqs: np array of length ``size of word vocabulary''
        super(MogFlow_batch, self).__init__()

        self.args = args

        emb_dim = self.emb_dim = args.emb_dim
        # variance not standard deviation!
        self.variance = var
        self.W = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        nn.init.orthogonal_(self.W)

    def cal_var(self, x, exp=True):
        # x: (batch_size, dim) -> the embedding of the base language
        # return var: (batch_size, dim)
        inpt_x = x
        for layer in self.log_var_nn:
            inpt_x = self.activation(layer(inpt_x))
        # clipping here
        inpt_x = inpt_x.clamp(max=7.0) * -1.0
        exp_inpt_x = torch.exp(inpt_x)
        self.var_analytics["max"] = exp_inpt_x.max().item()
        self.var_analytics["min"] = exp_inpt_x.min().item()
        self.var_analytics["mean"] = exp_inpt_x.mean().item()

        if exp:
            return torch.exp(inpt_x)
        else:
            return inpt_x

    def print_var_analytics(self):
        print(f"Max var = %.4f, Min var = %.4f, Mean var = %.4f" %
              (self.var_analytics["max"], self.var_analytics["min"], self.var_analytics["mean"]))

    def cal_mixture_of_gaussian(self, x_prime, x, x_freqs):
        # x_prime: (batch_size_1, dim) -> transformed space
        # x: (batch_size_2, dim) -> a sampled own language space
        # x_freq: (batch_size_2, )
        # calculate the base distribution p(x) = \sum_i p(x_i) * p(x|x_i) which is a mixture of gaussian
        # (x - x_i)^T(x - x_i) -> (batch_size_1, batch_size_2)
        log_variance = self.cal_var(x, exp=False) # (batch_size_2, dim)
        d = (((x_prime.unsqueeze(1) - x.unsqueeze(0)).pow(2)) / torch.exp(log_variance).unsqueeze(0)).sum(dim=2)
        # log|Sigma| = log \prod sigma_i = \sum log sigma_i
        logLL = -0.5 * (log_variance.sum(dim=1).unsqueeze(0) + d + self.emb_dim * math.log(2 * math.pi))
        logprob = x_freqs.unsqueeze(0) + logLL
        # (batch_size, )
        logLL = logsumexp(logprob, dim=1)
        return logLL

    def cal_mixture_of_gaussian_fix_var(self, x_prime, x, x_freqs, var=None, x_prime_freqs=None):
        # x_prime: (batch_size, dim) -> transformed space
        # x: (base_batch_size, dim) -> a sampled own language space
        # x_freq: (base_batch_size, ) # log freqs
        # calculate the base distribution p(x) = \sum_i p(x_i) * p(x|x_i) which is a mixture of gaussian
        # (x - x_i)^T(x - x_i) -> (batch_size, base_batch_size)
        # x_prime = x_prime / x_prime.norm(2, 1, keepdim=True)

        def _l2_distance(x, y):
            # x: N, d, y: M, d
            # return N, M
            return torch.pow(x, 2).sum(1).unsqueeze(1) - 2 * torch.mm(x, y.t()) + torch.pow(y, 2).sum(1)

        if self.args.init_var:
            log_det = torch.log(self.variance).sum()
            d = _l2_distance(x_prime, x) / self.variance.unsqueeze(0)
        else:
            log_det = self.emb_dim * math.log(self.variance)
            d = _l2_distance(x_prime, x) / self.variance

        logLL = -0.5 * (log_det + d + self.emb_dim * math.log(2 * math.pi)) # (batch_size, base_batch_size)
        if self.args.cofreq:
            # x_freqs and x_prime_freqs are just ranks
            freq_weight = F.log_softmax(-torch.abs(torch.log(x_freqs.unsqueeze(0)) - torch.log(x_prime_freqs.unsqueeze(1)))
                                    / self.args.temp, dim=1)
            logprob = freq_weight + logLL
        else:
            # x_freqs are in log scale
            logprob = x_freqs.unsqueeze(0) + logLL
        # (batch_size, )
        logLL = logsumexp(logprob, dim=1)
        return logLL

    def clip_logvar(self, logvar):
        with torch.no_grad():
            logvar.clamp_(self.args.clip_min_log_var, self.args.clip_max_log_var)
            # logvar.clamp_(self.args.clip_min_log_var, 0.)

    @overrides
    def forward(self, x: torch.tensor):
        # For back-translation purpose only
        return x.mm(self.W.t())

    @overrides
    def backward(self, y: torch.tensor, x: torch.tensor=None, x_freqs: torch.tensor=None, require_log_probs=True, var=None, y_freqs=None):
        # from other language to this language
        x_prime = y.mm(self.W)
        if require_log_probs:
            assert x is not None, x_freqs is not None
            log_probs = self.cal_mixture_of_gaussian_fix_var(x_prime, x, x_freqs, var, x_prime_freqs=y_freqs)
            _, log_abs_det = torch.slogdet(self.W)
            log_probs = log_probs + log_abs_det
        else:
            log_probs = torch.tensor(0)
        return x_prime, log_probs