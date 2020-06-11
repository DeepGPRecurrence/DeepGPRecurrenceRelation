"""Experiment with RQ kernel: to show change $\alpha$ does not affect the condition
Plot in the paper:
    - Figure 9 c, d
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
from gpytorch.kernels import (RQKernel,
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
        'size': 24
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
            # dont not collect sample from intermediate layers in this experiment
            # if self.collect:
            #     self.collector[d] = x.detach().numpy()

        # last layer with single output
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mvn = MultivariateNormal(mean_x, covar_x)
        x = mvn.sample(sample_shape=torch.Size([self.dim]))
        x = x.t()
        if self.collect:
            self.collector[self.depth - 1] = x.detach().numpy()
        return x


def create_rq(sigma2, lengthscale, alpha):
    rq = RQKernel()
    rq.lengthscale = lengthscale
    rq.alpha = alpha
    kernel = ScaleKernel(rq)
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


def experiment_rq_1d(save_dir, alpha, generate=True):
    n_data = 50
    x = torch.linspace(-5, 5, n_data)
    n_sample = 10
    depth = 50
    n_sigma = 20
    n_lengthscale = 20

    inv_lengthscale2s = np.linspace(0.1, 5, n_lengthscale)  # 1/p^2
    sigma2s = np.linspace(0.1, 5, n_sigma)

    if not generate:
        rmsd_matrix = np.zeros((n_lengthscale, n_sigma))

    inv_lengthscale2_iters = tqdm.tqdm(inv_lengthscale2s, desc='Inv Lengthscale2')
    for i, il2 in enumerate(inv_lengthscale2_iters):
        lengthscale = 1. / np.sqrt(il2)
        sigma2_iters = tqdm.tqdm(sigma2s, desc='Sigma2', leave=False)
        for j, sigma2 in enumerate(sigma2_iters):
            # save file is identify by indices, not by value
            if alpha == 0.5:
                # hard coded! Don't want to rerun sampling
                save_file = "inv_lengthscale2_{}_{}_sigma2_{}_{}.pkl".format(i, n_lengthscale, j, n_sigma)
            else:
                save_file = "inv_lengthscale2_{}_{}_sigma2_{}_{}_alpha_{}.pkl".format(i, n_lengthscale, j, n_sigma,
                                                                                      alpha)
            save_file = os.path.join(save_dir, save_file)
            if generate:
                kernel = create_rq(sigma2=sigma2,
                                   lengthscale=lengthscale,
                                   alpha=alpha)
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
                sigma2_iters.set_postfix_str(ret_str)
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
        x, y = np.meshgrid(inv_lengthscale2s, sigma2s)
        fig, ax = plt.subplots(figsize=(5, 5))
        CS = ax.contour(x, y, rmsd_matrix)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_xlabel(r'$1/\ell^2$', fontdict=font)
        ax.set_ylabel(r'$\sigma^2$', fontdict=font)
        ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
        ax.set_title(r'$\alpha={}$'.format(alpha))
        plt.savefig("../figure/experiment/rq_alpha_{}_1d.png".format(alpha), dpi=300, bbox_inches='tight')
        plt.show()


alpha = 0.5
# experiment_rq_1d("../data/exp_rq_1d", alpha, generate=True)
experiment_rq_1d("../data/exp_rq_1d", alpha, generate=False)

alpha = 3
# experiment_rq_1d("../data/exp_rq_1d", alpha, generate=True)
experiment_rq_1d("../data/exp_rq_1d", alpha, generate=False)
