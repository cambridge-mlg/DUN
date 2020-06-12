import torch
import torch.nn.functional as F
import torch.nn as nn


class res_DropoutBlock(nn.Module):
    """Skippable MLPBlock with relu"""
    def __init__(self, width, p_drop=0.5):
        super(res_DropoutBlock, self).__init__()
        self.p_drop = p_drop
        self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))

    def forward(self, x):
        return x + F.dropout(self.block(x), p=self.p_drop, training=True)


class DropoutBlock(nn.Module):
    """MLPBlock with relu"""
    def __init__(self, width_in, width_out, p_drop=0.5):
        super(DropoutBlock, self).__init__()
        self.p_drop = p_drop
        self.block = nn.Sequential(nn.Linear(width_in, width_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return F.dropout(self.block(x), p=self.p_drop, training=True)


class dropout_regression_homo(nn.Module):
    def __init__(self, input_dim, output_dim, width, n_layers, p_drop):
        super(dropout_regression_homo, self).__init__()

        self.p_drop = p_drop
        # self.log_std = nn.Parameter(torch.zeros(self.output_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.n_layers = n_layers

        self.layers = []
        self.layers.append(DropoutBlock(input_dim, width, self.p_drop))
        for i in range(self.n_layers - 1):
            self.layers.append(res_DropoutBlock(width, self.p_drop))
        self.layers += [nn.Linear(width, output_dim)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x  # , self.log_std.exp()

    def forward_predict(self, x, Nsamples=3):
        """This function is different from forward to compactly represent eval functions"""
        mu_vec = []
        for _ in range(Nsamples):
            x1 = self.layers(x)
            mu_vec.append(x1.data)
        mu_vec = torch.stack(mu_vec, dim=0)
        model_std = mu_vec.std(dim=0)
        # total_std = (self.log_std.exp()**2 + model_var).pow(0.5)
        mean = mu_vec.mean(dim=0)
        if Nsamples == 0:
            model_std = torch.zeros_like(mean)
        return mean, model_std

    def get_regulariser(self):
        """MC dropout uses weight decay to approximate the KL divergence"""
        return 0
