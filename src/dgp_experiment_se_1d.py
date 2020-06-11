"""Experiment with SE kernel 1 dimensional: trace of RMSDs
Plot in the paper:
    - Figure 8 (left)
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

sns.set()

# plot setup
matplotlib.rcParams.update({
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


def experiment_se_1d(save_dir, sigma2=1., generate=True):
    n_data = 50
    x = torch.linspace(-5, 5, n_data)
    n_sample = 30
    depth = 100

    ratios = np.array([0.1, 0.5, 0.8, 1, 1.1, 2, 3, 4, 5])

    if not generate:
        depth_aggs = []
        rmsd_aggs = []
        ratio_aggs = []

    ratio_iters = tqdm.tqdm(ratios, desc='Lengthscale')
    for ratio in ratio_iters:
        save_file = "r_{}_sigma2_{}.pkl".format(ratio, sigma2)
        save_file = os.path.join(save_dir, save_file)
        lengthscale = np.sqrt(sigma2 / ratio)
        if generate:
            kernel = create_rbf(sigma2, lengthscale=lengthscale)
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
            ratio_iters.set_postfix_str(ret_str)
        else:
            if not os.path.exists(save_file):
                print("{} is NOT existed. Generate sample first!!!".format(save_file))
                return
            collector = Collector(num_item=depth)
            collector.load(save_file)
            for d in range(depth - 1):
                samples = np.hstack(collector[d]).transpose()
                assert samples.shape[0] == n_sample and samples.shape[1] == n_data
                rmsds = post_process(samples)
                rmsd_aggs.extend(list(rmsds.flatten()))
                depth_aggs.extend([d] * len(rmsds))
                ratio_aggs.extend([ratio] * len(rmsds))

    if not generate:
        d = {"Depth": depth_aggs, "RMSD": rmsd_aggs, "Ratio": ratio_aggs}
        df = pd.DataFrame(data=d)
        ax = sns.relplot(x='Depth', y="RMSD", hue='Ratio', kind="line", legend='full', data=df)
        leg = ax._legend
        leg.set_bbox_to_anchor([0.99, 0.7])  # coordinates of lower left of bounding box
        leg.texts[0].set_text(r'$\sigma^2/\ell^2$')
        plt.title(r"$\sigma^2={}$".format(sigma2))
        plt.savefig("../figure/experiment/se_1d.png", dpi=300)
        plt.show()


experiment_se_1d(save_dir='../data/exp_se_1d', generate=True)
experiment_se_1d(save_dir='../data/exp_se_1d', generate=False)
