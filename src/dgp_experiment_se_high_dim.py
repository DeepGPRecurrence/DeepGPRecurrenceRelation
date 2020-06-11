"""Experiment settings: SE kernel, vary ratio $\ell^2/\sigma^2$ and vary dimesion $m$
Plot in the paper:
    - Figure 16
    - Figure 4c
"""
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;
import torch
import tqdm
from gpytorch import Module
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (RBFKernel, ScaleKernel)
from gpytorch.means import ZeroMean
from matplotlib import rc
from scipy.spatial.distance import cdist

sns.set(font_scale=2.5)

# plot setup
matplotlib.rcParams.update({
    'figure.figsize': (5, 5),
    'font.size': 18,
    'figure.subplot.bottom': 0.125,
})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18,
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


def create_rbf(sigma2, lengthscale):
    rbf = RBFKernel()
    rbf.lengthscale = lengthscale
    kernel = ScaleKernel(rbf)
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


def se_high_dim_from_recur(sigma2=1.):
    inv_ratios = np.arange(1, 11)  # lengthscale^2 / sigma^2
    dims = np.arange(1, 11)

    def recur(x, sigma2, lengthscale2, dim):
        half_dim = dim / 2.
        ret = 2 * sigma2 * dim * (1. - 1. / np.power(1 + x / (dim * lengthscale2), half_dim))
        return ret

    ratio_aggs = []
    ez_aggs = []
    dim_aggs = []

    inv_ratio_iters = tqdm.tqdm(inv_ratios, desc='Inv Ratio')
    for inv_ratio in inv_ratio_iters:
        lengthscale2 = sigma2 * inv_ratio
        dim_iters = tqdm.tqdm(dims, desc="Dimension", leave=False)
        for dim in dim_iters:
            x_init = 0.1
            x = x_init
            for _ in range(100):
                temp = recur(x, sigma2, lengthscale2, dim)
                x = temp

            ez = x
            ez_aggs.append(ez)
            dim_aggs.append(dim)
            ratio_aggs.append(inv_ratio)

    d = {"Ratio": ratio_aggs, "EZ": ez_aggs, "Dimension": dim_aggs}
    df = pd.DataFrame(data=d)
    ax = sns.relplot(x="Ratio", y="Dimension", size="EZ",
                     sizes=(40, 800),
                     height=10,
                     aspect=5 / 4,
                     data=df)
    ax.set_xlabels(r'$\ell^2/\sigma^2$')
    ax.set_ylabels(r'$m$')
    # ax.add_legend();
    leg = ax._legend
    leg.texts[0].set_text(r'$E[Z_l]$')
    for t in leg.texts[1:]:
        # truncate label text to 4 characters
        t.set_text(t.get_text()[:4])
    leg.set_bbox_to_anchor([1.05, 0.7])
    # leg._loc = 5
    # leg._borderaxespad = 1.2
    # ax._legend_out = True
    ax.fig.savefig("../figure/experiment/exp_se_recur_high_dim.png",
                   dpi=300,
                   bbox_extra_artists=(leg,),
                   bbox_inches='tight',
                   pad_inches=1.
                   )
    plt.show()


def experiment_se_high_dim(save_dir, sigma2=1., generate=True):
    n_data = 50
    x = torch.linspace(-5, 5, n_data)
    n_sample = 30
    depth = 100

    inv_ratios = np.arange(1, 11)  # Mixture scale2 / sigma^2
    dims = np.arange(1, 11)

    if not generate:
        ratio_aggs = []
        rmsd_aggs = []
        dim_aggs = []

    inv_ratio_iters = tqdm.tqdm(inv_ratios, desc='Inv Ratio')
    for inv_ratio in inv_ratio_iters:
        lengthscale = np.sqrt(sigma2 / inv_ratio)
        dim_iters = tqdm.tqdm(dims, desc="Dimension", leave=False)
        for dim in dim_iters:
            save_file = "inv_r_{}_dim_{}_sigma2_{}.pkl".format(inv_ratio, dim, sigma2)
            save_file = os.path.join(save_dir, save_file)

            if generate:
                kernel = create_rbf(sigma2, lengthscale=lengthscale)
                dgp = DeepGP(kernel,
                             depth=depth,
                             dim=dim)
                if os.path.exists(save_file):
                    ret_str = "{} is already existed. Skip!!!".format(save_file)
                else:
                    for _ in range(n_sample):
                        # collector will gather all samples
                        dgp(x)
                    collector = dgp.collector
                    collector.dump(save_file)
                    ret_str = "Write {}".format(save_file)
                dim_iters.set_postfix_str(ret_str)
            else:
                if not os.path.exists(save_file):
                    print("{} is NOT existed. Generate sample first!!!".format(save_file))
                    return
                collector = Collector(num_item=depth)
                collector.load(save_file)
                # choose a large depth
                selected_depth = depth - 1
                samples = collector[selected_depth]
                selected_dim = 0
                samples = [s[:, selected_dim] for s in samples]
                samples = np.vstack(samples)
                assert samples.shape[0] == n_sample and samples.shape[1] == n_data
                rmsds = post_process(samples)
                mean_rmds = np.mean(rmsds).squeeze()
                rmsd_aggs.append(mean_rmds)
                ratio_aggs.append(inv_ratio)
                dim_aggs.append(dim)

    if not generate:
        d = {"Ratio": ratio_aggs, "RMSD": rmsd_aggs, "Dimension": dim_aggs}
        df = pd.DataFrame(data=d)
        ax = sns.relplot(x="Ratio", y="Dimension", size="RMSD",
                         sizes=(40, 800),
                         height=10,
                         aspect=5 / 4,
                         data=df)
        ax.set_xlabels(r'$\ell^2/\sigma^2$')
        # ax.add_legend();
        leg = ax._legend
        for t in leg.texts:
            # truncate label text to 4 characters
            t.set_text(t.get_text()[:4])
        leg.set_bbox_to_anchor([0.85, 0.7])
        # leg._loc = 5
        # leg._borderaxespad = 1.2
        # ax._legend_out = True
        ax.fig.savefig("../figure/experiment/exp_se_high_dim.png",
                       dpi=300,
                       bbox_extra_artists=(leg,),
                       bbox_inches='tight',
                       pad_inches=1.
                       )
        plt.show()


## Figure 16
# experiment_se_high_dim(save_dir='../data/exp_se_high_dim', generate=True)
# experiment_se_high_dim(save_dir='../data/exp_se_high_dim', generate=False)

## Figure 4c
se_high_dim_from_recur()
