import torch
import torch.nn as nn

from src.DUN.layers import bern_MLPBlock, bern_MLPBlock_nores


class arq_uncert_fc_resnet(nn.Module):
    def __init__(self, input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False):
        super(arq_uncert_fc_resnet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, width)
        self.output_layer = nn.Linear(width, self.output_dim)
        self.n_layers = n_layers
        self.width = width
        self.w_prior = w_prior
        self.BMA_prior = BMA_prior
        if not isinstance(self.w_prior, list) and self.w_prior is not None:
            self.w_prior = [self.w_prior for i in range(self.n_layers + 1)]

        self.stochstic_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.stochstic_layers.append(bern_MLPBlock(width))

    def apply_prior(self, module, prior_f):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            self.loglike = self.loglike + prior_f(module.weight)
            self.loglike = self.loglike + prior_f(module.bias)

    def get_w_prior_loglike(self, k=None):
        if k is None:
            k = self.n_layers

        if self.w_prior is not None:
            self.loglike = 0
            self.input_layer.apply(lambda m: self.apply_prior(m, self.w_prior[0]))
            self.output_layer.apply(lambda m: self.apply_prior(m, self.w_prior[0]))
            self.loglikes = [self.loglike]  # append 0 for layer 0 and then remove it if necesary with BMA prior

            for idx, layer in enumerate(self.stochstic_layers[:k]):
                self.loglike = 0
                layer.apply(lambda m: self.apply_prior(m, self.w_prior[idx + 1]))
                self.loglikes.append(self.loglike)

            loglike_vec = torch.stack(self.loglikes, dim=0)

            if self.BMA_prior:
                loglike_vec = loglike_vec.unsqueeze(0).repeat(k+1, 1).tril(diagonal=0).sum(dim=1)  # size(depth)
            return loglike_vec.sum(dim=0)

        return self.input_layer.bias.data.new_zeros(k+1)

    def vec_forward(self, x, vec):
        assert vec.shape[0] == self.n_layers
        x = self.input_layer(x)
        for i in range(self.n_layers):
            x = self.stochstic_layers[i](x, vec[i])
        x = self.output_layer(x)
        return x

    def forward(self, x, depth=None):
        depth = self.n_layers if depth is None else depth
        act_vec = torch.zeros(depth+1, x.shape[0], self.output_dim).type(x.type())
        x = self.input_layer(x)
        act_vec[0] = self.output_layer(x)
        for i in range(depth):
            x = self.stochstic_layers[i](x, 1)
            act_vec[i+1] = self.output_layer(x)
        return act_vec


class arq_uncert_fc_MLP(arq_uncert_fc_resnet):
    def __init__(self, input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False):
        super(arq_uncert_fc_MLP, self).__init__(input_dim, output_dim, width, n_layers,
                                                w_prior=w_prior, BMA_prior=BMA_prior)

        self.stochstic_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.stochstic_layers.append(bern_MLPBlock_nores(width))
