import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD


class BayesLinear_local_reparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """
    def __init__(self, n_in, n_out):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.01, 0.01))
        self.W_p = nn.Parameter(
            torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(1, self.n_out).uniform_(-0.01, 0.01))
        self.b_p = nn.Parameter(torch.Tensor(1, self.n_out).uniform_(-3, -2))

    def forward(self, X):
        # calculate std
        std_w = F.softplus(self.W_p)
        std_b = F.softplus(self.b_p)

        act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
        act_W_std = torch.sqrt(1e-4 + torch.mm(X.pow(2), std_w.pow(2)))

        # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
        # the same random sample is used for every element in the minibatch output
        eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
        eps_b = Variable(self.b_mu.data.new(X.shape[0], std_b.shape[1]).normal_(mean=0, std=1))

        act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
        act_b_out = self.b_mu + std_b * eps_b

        output = act_W_out + act_b_out
        return output

    def get_KL(self, mu_p=0, sig_p=1):
        std_w = F.softplus(self.W_p)
        std_b = F.softplus(self.b_p)
        kld = KLD_cost(mu_p=mu_p, sig_p=sig_p, mu_q=self.W_mu, sig_q=std_w) \
            + KLD_cost(mu_p=mu_p, sig_p=sig_p, mu_q=self.b_mu, sig_q=std_b)
        return kld


class res_LocalReparamBlock(nn.Module):
    """Skippable MLPBlock with relu"""
    def __init__(self, width):
        super(res_LocalReparamBlock, self).__init__()
        self.block = nn.Sequential(BayesLinear_local_reparam(width, width), nn.ReLU(inplace=True))

    def forward(self, x):
        return x + self.block(x)

    def get_KL(self, mu_p=0, sig_p=1):
        return self.block[0].get_KL(mu_p=mu_p, sig_p=sig_p)


class LocalReparamBlock(nn.Module):
    """MLPBlock with relu"""
    def __init__(self, width_in, width_out):
        super(LocalReparamBlock, self).__init__()
        self.block = nn.Sequential(BayesLinear_local_reparam(width_in, width_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

    def get_KL(self, mu_p=0, sig_p=1):
        return self.block[0].get_KL(mu_p=mu_p, sig_p=sig_p)


class MFVI_regression_homo(nn.Module):
    def __init__(self, input_dim, output_dim, width, n_layers, prior_sig=1):
        super(MFVI_regression_homo, self).__init__()

        # self.log_std = nn.Parameter(torch.zeros(self.output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.n_layers = n_layers

        self.prior_sig = prior_sig
        if not isinstance(self.prior_sig, list):
            self.prior_sig = [self.prior_sig for i in range(self.n_layers + 1)]

        self.layers = []
        self.layers.append(LocalReparamBlock(input_dim, width))
        for i in range(self.n_layers - 1):
            self.layers.append(res_LocalReparamBlock(width))
        self.layers.append(BayesLinear_local_reparam(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, Nsamples=3):
        pred_vec = []
        for _ in range(Nsamples):
            x1 = self.layers(x)
            pred_vec.append(x1)
        pred_vec = torch.stack(pred_vec, dim=0)
        return pred_vec  # , self.log_std.exp()

    def forward_predict(self, x, Nsamples=3):
        """This function is different from forward to compactly represent eval functions"""
        mu_vec = []
        for i in range(Nsamples):
            x1 = self.layers(x)
            mu_vec.append(x1.data)
        mu_vec = torch.stack(mu_vec, dim=0)
        model_std = mu_vec.std(dim=0)
        
        # total_std = (self.log_std.exp()**2 + model_var).pow(0.5)
        mean = mu_vec.mean(dim=0)
        if Nsamples == 0:
            model_std = torch.zeros_like(mean)
        return mean, model_std

    def get_KL(self):
        KL = 0
        for layer_idx, layer in enumerate(self.layers):
            KL = KL + layer.get_KL(mu_p=0, sig_p=self.prior_sig[layer_idx])
        return KL
