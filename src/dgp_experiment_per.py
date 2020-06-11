"""Experiment with periodic kernel: contour plot of RMSD at layer 100
Plot in the paper:
    - Figure 9 a
"""

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from gpytorch import Module
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (PeriodicKernel,
                              ScaleKernel)
from gpytorch.means import ZeroMean
from matplotlib import rc
from scipy.spatial.distance import cdist

# plot setup
matplotlib.rcParams.update({
    'font.size': 18,
    'figure.subplot.bottom': 0.125,
})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 24,
        }


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
        x = mvn.sample(sample_shape=torch.Size([self.dim]))
        x = x.t()
        if self.collect:
            self.collector[self.depth - 1] = x.detach().numpy()
        return x


def create_per(sigma2, lengthscale, period_length):
    per = PeriodicKernel()
    per.lengthscale = lengthscale
    per.period_length = period_length
    kernel = ScaleKernel(per)
    kernel.outputscale = sigma2
    return kernel


def compute_rmsd(x):
    n = x.shape[0]
    sd = cdist(x, x)
    den = n * (n - 1)
    sum_sd = np.sum(sd)
    return np.sqrt(sum_sd / den)


def post_process(samples):
    n_sample = samples.shape[0]
    rmsd = np.zeros(n_sample)
    for i in range(n_sample):
        s = samples[i][:, None]
        rmsd_i = compute_rmsd(s)
        rmsd[i] = rmsd_i
    return rmsd


def experiment_per_1d(save_dir, generate=True):
    """
    Experiment with periodic kernel
    :param save_dir: save directory
    :param generate: if we generate data or we plot figure from data
    """
    n_data = 50
    x = torch.linspace(-5, 5, n_data)
    n_sample = 30
    depth = 100
    sigma2 = 1.
    n_per = 10
    n_lengthscale = 10
    inv_per2 = np.linspace(0.2, 3, n_per)  # 1/p^2
    lengthscales = np.linspace(0.8, 5, n_lengthscale)

    if not generate:
        rmsd_matrix = np.zeros((n_per, n_lengthscale))

    inv_per2_iters = tqdm.tqdm(inv_per2, desc='Period')
    for i, ip2 in enumerate(inv_per2_iters):
        p = 1. / np.sqrt(ip2)
        lengthscale_iters = tqdm.tqdm(lengthscales, desc='Lengthscale', leave=False)
        for j, lengthscale in enumerate(lengthscale_iters):
            # save file is identify by indices, not by value
            save_file = "inv_per2_{}_{}_lengthscale_{}_{}.pkl".format(i, n_per, j, n_lengthscale)
            save_file = os.path.join(save_dir, save_file)
            if generate:
                kernel = create_per(sigma2,
                                    lengthscale=lengthscale,
                                    period_length=p)
                dgp = DeepGP(kernel,
                             depth=depth,
                             dim=1)
                if os.path.exists(save_file):
                    ret_str = "{} is already existed. Skip!!!".format(save_file)
                else:
                    for _ in range(n_sample):
                        # collector will gather all samples
                        dgp(x)
                    collector = dgp.collector
                    collector.dump(save_file)
                    ret_str = "Write {}".format(save_file)
                lengthscale_iters.set_postfix_str(ret_str)
            else:
                if not os.path.exists(save_file):
                    print("{} is NOT existed. Generate sample first!!!".format(save_file))
                    return
                collector = Collector(num_item=depth)
                collector.load(save_file)
                # choose a large depth
                selected_depth = depth - 1
                samples = np.hstack(collector[selected_depth]).transpose()
                assert samples.shape[0] == n_sample and samples.shape[1] == n_data
                rmsds = post_process(samples)
                rmsd_matrix[i, j] = rmsds.mean()

    if not generate:
        x, y = np.meshgrid(inv_per2, lengthscales)
        fig, ax = plt.subplots(figsize=(5, 5))
        CS = ax.contour(x, y, rmsd_matrix)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_xlabel(r'$1/p^2$', fontdict=font)
        ax.set_ylabel(r'$\ell$', fontdict=font)
        ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)

        plt.savefig("../figure/experiment/per_1d.png", dpi=300, bbox_inches='tight')
        plt.show()


experiment_per_1d("../data/exp_per_1d", generate=True)
experiment_per_1d("../data/exp_per_1d", generate=False)
