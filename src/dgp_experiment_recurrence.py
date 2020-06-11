"""This script is to justify the recurrence of the expectation
Kernels: SE and SM

Plot in the papers:
    - Figure 7
    - Figure 17
    - Figure 18.
"""

import math
import pickle

import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (RBFKernel, SpectralMixtureKernel, Kernel)
from gpytorch.means import ZeroMean
from matplotlib import rc

# plot setup
sns.set()
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


class DeepGP(gpytorch.Module):

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


def recur(x, sigma2, lengthscale2, dim):
    half_m = 1. * dim / 2
    return 2 * sigma2 * (1. - 1. / np.power(1 + x / (lengthscale2), half_m))


def sm_moment_generating(t, lambda_term, dim):
    den = 1 - 2 * t
    exp_term = np.exp(lambda_term * t / den)
    return exp_term / np.sqrt(den ** dim)


def sm_moment_generating_1d(t, lambda_term):
    return sm_moment_generating(t, lambda_term, dim=1)


def recur_sm_final(x, sigma, mu, dim):
    mu2 = mu ** 2
    sigma2 = sigma ** 2
    lambda_term = - mu2 / (4 * np.pi ** 2 * sigma2 ** 2 * x)
    t = - 2 * np.pi ** 2 * sigma2 * x
    exp_term = np.exp(-0.5 * mu2 / sigma2)
    EZ = exp_term * sm_moment_generating_1d(t, lambda_term)
    return 2 * (1 - EZ ** dim)


def recur_sm_final_2(x, sigma, mu, dim):
    mu2 = mu ** 2
    sigma2 = sigma ** 2
    pi2 = np.pi ** 2
    den = 1. + 4 * pi2 * sigma2 * x
    exp_term_1 = np.exp(-mu2 / (2 * sigma2))
    exp_term_2 = np.exp(dim * mu2 / (2 * sigma2 * den))
    return 2 * (1 - exp_term_1 * exp_term_2 * np.power(den, -0.5 * dim))


def recur_sm_2(x, lengthscale, p, m):
    ell2 = lengthscale ** 2
    p2 = p ** 2
    sigma2 = 0.5 * np.exp(-0.5 * np.pi ** 2 * ell2 / p2)
    v2 = 0.5 / ell2
    u = np.pi * ell2 / p
    u2 = u ** 2
    t = - x * v2
    lambda_term = -u2 / (x)
    mt = sm_moment_generating(t, lambda_term, m)
    return 2 * (1. - 2 * sigma2 * mt)


def recur_sm(x, lengthscale, p, m):
    ell2 = lengthscale ** 2
    p2 = p ** 2
    den = 1 + x / ell2
    first_exp = np.exp(np.pi ** 2 * ell2 / (2 * p2))
    second_exp = np.exp(-np.pi ** 2 * ell2 * x / (2 * p2 * den))
    ret = 2 * (1 - first_exp * second_exp * np.power(den, -m / 2.))
    return ret


def tracking_expectation(ratio, dim, n_sample=2000, depth=11):
    """

    :param ratio: \sigma^2 / lengthscale^2
    :param dim:
    :param n_sample:
    :param depth:
    :return:
    """
    sigma2 = 1.  # default
    lengthscale2 = sigma2 / ratio

    kernel = RBFKernel()
    kernel.lengthscale = np.sqrt(lengthscale2)

    # x contains two data points
    x = torch.from_numpy(np.array([0., 1.]).astype('float32')).view(2, 1)
    # build model
    dgp = DeepGP(kernel, depth=depth, dim=dim)

    ## compute true expectation via recurrence
    Kx = kernel(x).evaluate_kernel()
    k12 = Kx.detach().numpy()[0, 1]
    EZ_1 = 2 * (1 - k12)
    true_EZ = [EZ_1]
    temp = EZ_1
    for d in range(1, depth - 1):
        next_EZ = recur(temp, sigma2, lengthscale2, dim=dim)
        true_EZ.append(next_EZ)
        temp = next_EZ

    # sample and collect
    for i in range(n_sample):
        dgp(x)
    collector = dgp.collector

    aggs_depth = []
    aggs_diff = []
    aggs_empirical = []
    # post-process sample
    for d in range(depth - 1):
        samples = collector[d]
        samples = np.stack(samples)  # n_sample x 2 x dim
        diff = np.diff(samples, axis=1)  # n_sample x dim
        diff = diff ** 2
        diff = diff.squeeze()
        # take 1 dimension
        selected_dim = 0
        diff = diff[:, selected_dim]
        aggs_depth.extend([d + 1] * n_sample)
        aggs_diff.extend(diff.tolist())
        aggs_empirical.extend(["empirical"] * n_sample)

    ## true expectation
    aggs_depth.extend(list(range(1, depth)))
    aggs_diff.extend(true_EZ)
    aggs_empirical.extend(["ours"] * len(true_EZ))

    ##
    ## THIS RESULT FROM DUNLOP. BUT CANNOT PLOT SINCE EZ IS TOO BIG
    # aggs_depth.extend(list(range(1, depth)))
    # aggs_diff.extend(dunlop_EZ)
    # aggs_empirical.extend(["Dunlop 2018"]*len(dunlop_EZ))

    d = {"Depth": aggs_depth, "Z": aggs_diff, "empirical": aggs_empirical}
    df = pd.DataFrame(data=d)
    ax = sns.pointplot(x='Depth', y='Z', hue='empirical', data=df, capsize=0.2, markers=["o", "*", "d"], join=False)
    ax.set_ylabel(r'Expectation of $Z_l$', fontdict=font)
    ax.set_xlabel(r'Layer $l$', fontdict=font)
    leg = ax.legend()
    ax.set_title(r"${{\sigma^2}}/{{\ell^2}}={}$,   $m={}$".format(ratio, dim), fontdict=font)
    name = "r_{}_m_{}".format(ratio, dim)
    plt.savefig("../figure/track_expectation/" + name + ".png", bbox_extra_artists=(leg,), bbox_inches='tight', dpi=300)


def tracking_expectation_sm(sigma2, mu, dim, n_sample=2000, depth=11):
    def create_sm_2(sigma2, mu, dim):
        sm = SpectralMixtureKernel(num_mixtures=1, ard_num_dims=dim)
        sm.mixture_weight = 1.
        sm.mixture_scales = sigma2  # rescale as in the paper
        sm.mixture_means = mu  # rescale as in the paper
        return sm

    def create_sm(sigma2, mu):
        sm = SMKernel_1D()
        sm.scale = sigma2
        sm.mean = mu
        return sm

    # kernel = create_sm_2(sigma2=np.sqrt(lengthscale2), mu=period_length, dim=dim)

    kernel = create_sm(sigma2=np.sqrt(sigma2), mu=mu)

    # x contains two data points
    x = torch.from_numpy(np.array([0., 1.]).astype('float32')).view(2, 1).repeat(1, dim)
    # build model
    dgp = DeepGP(kernel, depth=depth, dim=dim)

    ## compute true expectation via recurrence
    Kx = kernel(x).evaluate_kernel()
    k12 = Kx.detach().numpy()[0, 1]
    EZ_1 = 2 * (1 - k12)
    true_EZ = [EZ_1]
    temp = EZ_1
    for d in range(1, depth - 1):
        next_EZ = recur_sm_final(temp, sigma=np.sqrt(sigma2), mu=mu, dim=dim)
        # next_EZ_1 = recur_sm_final_2(temp, sigma=np.sqrt(sigma2), mu=mu, dim=dim)
        true_EZ.append(next_EZ)
        temp = next_EZ

    # # sample and collect
    for i in range(n_sample):
        dgp(x)
    collector = dgp.collector
    #
    aggs_depth = []
    aggs_diff = []
    aggs_empirical = []
    # # post-process sample
    for d in range(depth - 1):
        samples = collector[d]
        samples = np.stack(samples)  # n_sample x 2 x dim
        diff = np.diff(samples, axis=1)  # n_sample x dim
        diff = diff ** 2
        diff = diff.squeeze()
        if dim > 1:
            # take 1 dimension
            selected_dim = 0
            diff = diff[:, selected_dim]
        aggs_depth.extend([d + 1] * n_sample)
        aggs_diff.extend(diff.tolist())
        aggs_empirical.extend(["empirical"] * n_sample)

    ## true expectation
    aggs_depth.extend(list(range(1, depth)))
    aggs_diff.extend(true_EZ)
    aggs_empirical.extend(["true"] * len(true_EZ))

    d = {"Depth": aggs_depth, "Z": aggs_diff, "empirical": aggs_empirical}
    df = pd.DataFrame(data=d)
    ax = sns.pointplot(x='Depth', y='Z', hue='empirical', data=df, capsize=0.2, markers=["o", "*", "d"], join=False)
    ax.set_ylabel(r'Expectation of $Z_l$', fontdict=font)
    ax.set_xlabel(r'Layer $l$', fontdict=font)
    leg = ax.legend()
    ax.set_title(r'$\sigma^2={}, \mu = {}, m = {}$'.format(sigma2, mu, dim))
    name = "SM_{}_{}_m_{}".format(sigma2, mu, dim)
    plt.savefig("../figure/track_expectation/" + name + ".png", bbox_extra_artists=(leg,), bbox_inches='tight', dpi=300)


class SMKernel_1D(Kernel):
    """A simple implementation of spectral mixture kernel"""

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


### SE
## hyperparameter
# ratio = 2
# dim = 30
# tracking_expectation(ratio=ratio, dim=dim)


### SM
# sigma = 0.5
# mu = 0.5
# dim = 1
# tracking_expectation_sm(sigma, mu, dim=dim)


# sigma = 1.
# mu = 0.5
# dim = 1
# tracking_expectation_sm(sigma, mu, dim=dim)


# sigma = 1.
# mu = 1.
# dim = 1
# tracking_expectation_sm(sigma, mu, dim=dim)

sigma = 2
mu = 1.
dim = 1
tracking_expectation_sm(sigma, mu, dim=dim)

# sm = SMKernel()
# sm.scale = 1.
# sm.mean = 1.
# x = torch.linspace(0, 1, 4).view(2,2)
# K= sm(x).evaluate_kernel()

plt.show()
