"""Experiment with SM kernel in 1 dimensional case: contour plot of RMSDs
Plot in the paper:
    - Figure 9b
"""

import math
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from gpytorch import Module
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (Kernel)
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


class SMKernel_1D(Kernel):
    """A simplified version of spectral mixture kernel"""

    def __init__(self):
        super().__init__()

        ms_shape = torch.Size([1, 1])
        self.register_parameter(name="raw_scale", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))
        self.register_parameter(name="raw_mean", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))

        self.register_constraint("raw_scale", Positive())
        self.register_constraint("raw_mean", Positive())

    @property
    def scale(self):
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    @property
    def mean(self):
        return self.raw_mean_constraint.transform(self.raw_mean)

    @mean.setter
    def mean(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mean)
        self.initialize(raw_mean=self.raw_mean_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):

        # exp term
        x1_exp = x1 * self.scale * math.pi
        x2_exp = x2 * self.scale * math.pi
        exp_term = self.covar_dist(x1_exp, x2_exp, square_dist=True).mul_(-2).exp_()

        # cosine term
        x1_cos = x1 * self.mean * math.pi
        x2_cos = x2 * self.mean * math.pi

        cos_term = self.covar_dist(x1_cos, x2_cos).mul_(2).cos_()

        return exp_term * cos_term


def create_sm(sigma, mu):
    sm = SMKernel_1D()
    sm.scale = sigma
    sm.mean = mu
    return sm


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


def experiment_sm_1d(save_dir, generate=True):
    n_data = 50
    x = torch.linspace(-5, 5, n_data)
    n_sample = 30
    depth = 100
    n_sigma = 10
    n_mu = 10
    sigmas = np.linspace(0.1, 1, n_sigma)
    mus = np.linspace(0.1, 1, n_mu)

    if not generate:
        rmsd_matrix = np.zeros((n_sigma, n_mu))

    sigma_iters = tqdm.tqdm(sigmas, desc='Sigma')
    for i, sigma in enumerate(sigma_iters):
        mu_iters = tqdm.tqdm(mus, desc='Mu', leave=False)
        for j, mu in enumerate(mu_iters):
            # save file is identify by indices, not by value
            save_file = "sigma_{}_mu_{}.pkl".format(i, j)
            save_file = os.path.join(save_dir, save_file)
            if generate:
                kernel = create_sm(sigma=sigma, mu=mu)
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
                mu_iters.set_postfix_str(ret_str)
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
        x, y = np.meshgrid(sigmas, mus)
        fig, ax = plt.subplots(figsize=(5, 5))
        CS = ax.contour(x, y, rmsd_matrix)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_xlabel(r'$\sigma^2$', fontdict=font)
        ax.set_ylabel(r'$\mu$', fontdict=font)
        ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)

        plt.savefig("../figure/experiment/sm_1d.png", dpi=300, bbox_inches='tight')
        plt.show()


experiment_sm_1d("../data/exp_sm_1d", generate=True)
experiment_sm_1d("../data/exp_sm_1d", generate=False)
