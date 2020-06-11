"""Generate data for Figure 3"""
import pickle

import matplotlib.pyplot as plt
import torch
from gpytorch import Module
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (RBFKernel, ScaleKernel)
from gpytorch.means import ZeroMean
import tikzplotlib


class Collector(object):

    def __init__(self, num_item):
        self.store = {}
        for i in range(num_item):
            self.store[i] = []

    def __setitem__(self, key, value):
        self.store[key].append(value)

    def __getitem__(self, item):
        return self.store[item]

    def dump(self, save_file):
        with open(save_file, 'wb') as f:
            pickle.dump(self.store, f)

    def load(self, save_file):
        with open(save_file, 'rb') as f:
            self.store = pickle.load(f)


def create_rbf(sigma2, lengthscale):
    rbf = RBFKernel()
    rbf.lengthscale = lengthscale
    kernel = ScaleKernel(rbf)
    kernel.outputscale = sigma2
    return kernel


class DeepGP(Module):

    def __init__(self, kernel, depth, dim=1, collect=True):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.covar_module = kernel
        self.mean_module = ZeroMean()
        self.collect = collect
        if self.collect:
            self.collector = Collector(depth)

    def forward(self, x):
        for d in range(self.depth - 1):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            mvn = MultivariateNormal(mean_x, covar_x)
            x = mvn.sample(sample_shape=torch.Size([self.dim]))
            x = x.t()
            if self.collect:
                self.collector[d] = x.detach().numpy()

        # last layer with single output
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mvn = MultivariateNormal(mean_x, covar_x)
        x = mvn.sample()
        if self.collect:
            self.collector[self.depth - 1] = x.detach().numpy()
        return x


kernel = create_rbf(1., 0.5)
dgp = DeepGP(kernel, depth=3)
x = torch.linspace(0, 1, 50)
dgp(x)

collector = dgp.collector


def plot_sample(x, y, color='r', save_file="../figure/layer_l_1.tex"):
    plt.figure(figsize=(3, 3))
    ax1 = plt.axes(frameon=False)

    ax1.plot(x, y, color=color, lw=2)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    tikzplotlib.save(save_file)


# plt.plot(x, collector[1][0].squeeze())
# plt.plot(x, collector[2][0].squeeze())
# tikzplotlib.save("../figure/layer_l_1.tex")
plot_sample(x, collector[1][0].squeeze(), save_file="../figure/layer_l_1.tex")
plot_sample(x, collector[2][0].squeeze(), save_file="../figure/layer_l.tex")
# plt.show()
