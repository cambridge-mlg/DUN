import torch.nn as nn


class global_mean_pool_2d(nn.Module):
    def __init__(self):
        super(global_mean_pool_2d, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2, 3))


class res_MLPBlock(nn.Module):
    def __init__(self, width):
        super(res_MLPBlock, self).__init__()
        self.ops = nn.Sequential(nn.Linear(width, width), nn.ReLU(),  nn.BatchNorm1d(num_features=width))

    def forward(self, x):
        return x + self.ops(x)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class bern_MLPBlock(nn.Module):
    """Skippable MLPBlock with relu"""
    def __init__(self, width):
        super(bern_MLPBlock, self).__init__()

        self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(num_features=width))

    def forward(self, x, b):
        """b is sample from binary variable or activation probability (soft forward)"""
        return x + b * self.block(x)


class bern_MLPBlock_nores(nn.Module):
    """Skippable MLPBlock with relu"""
    def __init__(self, width):
        super(bern_MLPBlock_nores, self).__init__()
        self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(num_features=width))

    def forward(self, x, b):
        """b is sample from binary variable or activation probability (soft forward)"""
        return self.block(x)


class bern_leaky_MLPBlock(nn.Module):
    """Skippable MLPBlock with leaky relu"""
    def __init__(self, width):
        super(bern_leaky_MLPBlock, self).__init__()

        self.block = nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(), nn.BatchNorm1d(num_features=width))

    def forward(self, x, b):
        """b is sample from binary variable or activation probability (soft forward)"""
        return x + b * self.block(x)


class bern_preact_MLPBlock(nn.Module):
    """Skippable MLPBlock with preactivation"""
    def __init__(self, width):
        super(bern_preact_MLPBlock, self).__init__()

        self.block = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, width))

    def forward(self, x, b):
        """b is sample from binary variable or activation probability (soft forward)"""
        return x + b * self.block(x)


class bern_leaky_preact_MLPBlock(nn.Module):
    """Skippable MLPBlock with preactivation and leaky relu"""
    def __init__(self, width):
        super(bern_leaky_preact_MLPBlock, self).__init__()

        self.block = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, width))

    def forward(self, x, b):
        """b is sample from binary variable or activation probability (soft forward)"""
        return x + b * self.block(x)


class bern_basic_convBlock(nn.Module):
    """Skippable basic convolutional preactivation Resnet Block, now with optional downsampling!"""
    def __init__(self, num_channels, downsample=False):
        super(bern_basic_convBlock, self).__init__()

        first_stride = 2 if downsample else 1

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=first_stride, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_channels),
        )

    def forward(self, x, b):
        return x + b * self.block(x)


class bern_leaky_basic_convBlock(nn.Module):
    """Skippable basic convolutional preactivation Resnet Block, now with optional downsampling!"""
    def __init__(self, num_channels, downsample=False):
        super(bern_leaky_basic_convBlock, self).__init__()

        first_stride = 2 if downsample else 1

        self.block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=first_stride, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_channels),
        )

    def forward(self, x, b):
        return x + b * self.block(x)


class bern_bottleneck_convBlock(nn.Module):
    """Skippable bottleneck convolutional preactivation Resnet Block"""
    def __init__(self, inner_dim, outer_dim):
        super(bern_bottleneck_convBlock, self).__init__()

        self.block = self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, x, b):
        return x + b * self.block(x)


class bern_leaky_bottleneck_convBlock(nn.Module):
    """Skippable bottleneck convolutional preactivation Resnet Block"""
    def __init__(self, inner_dim, outer_dim):
        super(bern_leaky_bottleneck_convBlock, self).__init__()

        self.block = self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, x, b):
        return x + b * self.block(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
