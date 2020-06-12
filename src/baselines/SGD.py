import random
import numpy as np
import torch
import torch.nn as nn


class res_MLPBlock(nn.Module):
    """Skippable MLPBlock with relu"""
    def __init__(self, width):
        super(res_MLPBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(width))  # nn.LayerNorm(width))  # batch norm doesnt really make sense for MCMC

    def forward(self, x):
        """b is sample from binary variable or activation probability (soft forward)"""
        return x + self.block(x)


class SGD_regression_homo(nn.Module):
    def __init__(self, input_dim, output_dim, width, n_layers, seed=None):
        super(SGD_regression_homo, self).__init__()

        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # self.log_std = nn.Parameter(torch.zeros(self.output_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.n_layers = n_layers

        self.layers = []
        self.layers += [nn.Linear(input_dim, width), nn.ReLU(), nn.BatchNorm1d(width)]
        for _ in range(self.n_layers - 1):
            self.layers.append(res_MLPBlock(width))
        self.layers += [nn.Linear(width, output_dim)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        mean = self.layers(x)
        return mean # , self.log_std.exp()

    def forward_predict(self, x, Nsamples=0):
        """This function is different from forward to compactly represent eval functions"""
        mu = self.forward(x)
        return mu, torch.ones_like(mu) * 0 # TODO: torch.zeros_like?

    def get_regulariser(self):
        """MC dropout uses weight decay to approximate the KL divergence"""
        return 0
