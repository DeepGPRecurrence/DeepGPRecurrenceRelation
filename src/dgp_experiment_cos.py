"""Experiment with COSINE kernel: trace of RMSD
Plot in the paper:
    - Figure 8 (left)
"""

import torch
import numpy as np
import pickle
import os
import tqdm
import pandas as pd

from gpytorch import Module
from gpytorch.means import ZeroMean
from gpytorch.kernels import (RBFKernel, CosineKernel, PeriodicKernel,
                              ScaleKernel)
from gpytorch.distributions import MultivariateNormal
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import seaborn as sns; sns.set()

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


def create_cosine(sigma2, period_length):
    cosine = CosineKernel()
    cosine.period_length = period_length
    kernel = ScaleKernel(cosine)
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


def experiment_cosine_1d(save_dir, sigma2=1., generate=True, zoom=False, zoom_x = [], zoom_y = []):
    n_data = 50
    x = torch.linspace(-5, 5, n_data)
    n_sample = 30
    depth = 100

    # ratios = np.array([0.1, 0.5, 0.9, 1., 1.1, 1.2, 1.5, 2.])
    ratios = np.array([0.1, 0.5, 0.8, 1., 1.1, 2, 3, 4, 5])

    if not generate:
        depth_aggs = []
        rmsd_aggs = []
        ratio_aggs = []

    ratio_iters = tqdm.tqdm(ratios, desc='Ratio')
    for ratio in ratio_iters:
        save_file = "r_{}_sigma2_{}.pkl".format(ratio, sigma2)
        save_file = os.path.join(save_dir, save_file)
        p = np.sqrt(sigma2 / ratio) * np.pi
        if generate:
            kernel = create_cosine(sigma2, period_length=p)
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
            for d in range(depth-1):
                samples = np.hstack(collector[d]).transpose()
                assert samples.shape[0] == n_sample and samples.shape[1] == n_data
                rmsds = post_process(samples)
                rmsd_aggs.extend(list(rmsds.flatten()))
                depth_aggs.extend([d] * len(rmsds))
                ratio_aggs.extend([ratio] * len(rmsds))

    if not generate:
        d = {"Depth": depth_aggs, "RMSD": rmsd_aggs, "Ratio": ratio_aggs}
        df = pd.DataFrame(data=d)
        if not zoom:
            ax = sns.relplot(x='Depth', y="RMSD", hue='Ratio', kind="line", legend='full', data=df)
            leg = ax._legend
            leg.set_bbox_to_anchor([1.02, 0.7])  # coordinates of lower left of bounding box
            # leg._loc = 2
            leg.texts[0].set_text(r'$\pi^2 \sigma^2 / p^2$')
            plt.title(r'$\sigma^2={}$'.format(sigma2))
            plt.savefig("../figure/experiment/cosine_1d.png", dpi=300)
        else:
            ax = sns.relplot(x='Depth', y="RMSD", hue='Ratio', kind="line", legend=False, data=df)
            ax = ax.axes.flatten()[0]
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xlim(zoom_x)
            ax.set_ylim(zoom_y)
            plt.savefig("../figure/experiment/cosine_1d_zoom.png", dpi=300)
        plt.show()


experiment_cosine_1d("../data/exp_cosine_1d", generate=True)
experiment_cosine_1d("../data/exp_cosine_1d", generate=False)
# experiment_cosine_1d("../data/exp_cosine_1d", generate=False, zoom=True, zoom_x=[50, 60], zoom_y=[0,0.1])

