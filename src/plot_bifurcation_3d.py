"""This code is to plot of bifurcation and contour for recurrence relation
Plot in the papers:
    - Figure 4 a, b
    - Figure 5 a, b, c
    - Figure 11, 12, 13, 14
"""


import os
import pickle

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.pyplot import cm
from mpmath import hyp2f0

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


## Define recurrence relation
def recur_se(x, sigma2, rate):
    """Recurrence relation for SE kernel 1D case
        rate = 1/\ell^2
    """
    return 2 * sigma2 * (1. - 1. / np.sqrt(1 + rate * x))


def recure_se_high_dim(x, sigma2, rate, m):
    """Recurrence relation for SE kernel 1D case
            rate = 1/\ell^2
    """
    half_m = 0.5 * m
    return 2 * sigma2 * m * (1. - 1. / np.power(1 + rate * x / m, half_m))


def recur_rational_quadratic(x, sigma2, alpha, rate):
    """

    :param x:
    :param alpha: \alpha
    :param rate: 1/\ell^2
    :return:
    """
    z = - rate * x / (alpha)
    fz = hyp2f0(alpha, 0.5, z)
    return 2 * sigma2 * (1. - fz.real)

def recur_input_connected(x, sigma2, rate):
    """Recurrent relation for input connected DGP"""
    return 2 * sigma2 * (1. - 1. / np.sqrt(1 + rate * x)) + 0.5


def recur_cosin(x, sigma2, rate):
    """Recurrence relation for cosine kernel"""
    return 2 * sigma2 * (1. - np.exp(-rate * x/2))

def recur_cosin_standard(x, sigma2, rate):
    """Recurrence relation for cosine kernel"""
    return 2 * sigma2 * (1. - np.exp(-rate * np.pi**2 * x))

def recur_periodic(x, sigma2, inv_period2, inv_lengthscale2):
    """Recurrence relation for periodic kernel"""
    ret = 2 * sigma2 * inv_lengthscale2 * (1 - np.exp(-np.pi ** 2 * x * inv_period2))
    return ret

def recur_periodic_contour(x, inv_period2, lengthscale):

    """Function to plot contour for PER"""
    inv_lengthscale2 = 1./lengthscale**2
    return recur_periodic(x,
                          sigma2=1.,
                          inv_period2=inv_period2,
                          inv_lengthscale2=inv_lengthscale2)


def recur_sm(x, sigma2, mu, dim):
    """Recurrence relation of spectral mixture kernel"""
    mu2 = mu ** 2
    pi2 = np.pi **2
    den = 1. + 4 * pi2 * sigma2 * x
    exp_term_1 = np.exp(-mu2 / (2*sigma2))
    exp_term_2 = np.exp(dim * mu2 /(2* sigma2 * den))
    return 2*(1 - exp_term_1 * exp_term_2 * np.power(den, -0.5 * dim))

def recur_sm_1d(x, sigma2, mu):
    """Recurrence relation of spectral mixture kernel for 1D case"""
    return recur_sm(x, sigma2, mu, 1)


def generate_recur_2d(recur, sigma, num_discard=500, num_gen=200, min_rate=0., max_rate=5.0, num_rate=500):
    """Generate data to create bifurcation plot"""
    rates = np.linspace(min_rate, max_rate, num_rate)
    x_inital = 0.1
    X, Y = [], []


    for rate in rates:
        x = x_inital
        # burn-in phase
        for _ in range(num_discard):
            z = recur(x, sigma, rate)
            x = z

        # collected valued
        for _ in range(num_gen):
            z = recur(x, sigma, rate)
            x = z
            X.append(rate)
            Y.append(z)

    return X, Y


def generate_recur_rq(sigma, alpha, num_discard=500, num_gen=200, min_rate=0., max_rate=5.0, num_rate=500):
    rates = np.linspace(min_rate, max_rate, num_rate)
    x_inital = 0.1
    X, Y = [], []

    # burn-in phase
    for rate in rates:
        x = x_inital
        for _ in range(num_discard):
            z = recur_rational_quadratic(x, sigma, alpha, rate)
            x = z

        for _ in range(num_gen):
            z = recur_rational_quadratic(x, sigma, alpha, rate)
            x = z
            X.append(rate)
            Y.append(z)

    return X, Y


def generate_recur_per(sigma, inv_lengthscal2, num_discard=300, num_gen=200, min_rate=0., max_rate=5.0, num_rate=300):
    rates = np.linspace(min_rate, max_rate, num_rate)
    x_inital = 0.1
    X, Y = [], []

    # burn-in phase
    for rate in rates:
        x = x_inital
        for _ in range(num_discard):
            z = recur_periodic(x, sigma2=sigma,
                               inv_period2=rate,
                               inv_lengthscale2=inv_lengthscal2)
            x = z

        for _ in range(num_gen):
            z = recur_periodic(x, sigma2=sigma,
                               inv_period2=rate,
                               inv_lengthscale2=inv_lengthscal2)
            x = z
            X.append(rate)
            Y.append(z)

    return X, Y


def generate_recur_3d(recur, num_discard=300,
                      num_gen=200,
                      min_x=0.,
                      max_x=5.,
                      min_y=0.,
                      max_y=5.,
                      num_x=100,
                      num_y=100,
                      Xs=None):
    if Xs is None:
        Xs = np.linspace(min_x, max_x, num_x)
    Ys = np.linspace(min_y, max_y, num_y)
    u_0 = 0.1
    X_Y, Us = [], []

    for x in Xs:
        for y in Ys:
            u = u_0
            # burn-in phase
            for _ in range(num_discard):
                temp = recur(u, x, y)
                u = temp
            # record the value
            for _ in range(num_gen):
                temp = recur(u, x, y)
                u = temp
                X_Y.append((x, y))
                Us.append(u)
    return X_Y, Us


def plot_chaotic(name, recur=recur_se, figsize=(5, 5), xlabel=r"$\frac{1}{\ell^2}$", ylabel_fm=r'$\sigma^2={}$',
                 yticks=None, xticks = [0.5, 1, 2.5, 3]):
    # picking sigma2 = 0.1,0.5, 1.,2.,4., 6.
    Xs = np.array([0.33, 0.5, 1., 2., 4., 6.])

    # choose color map
    colormap = cm.viridis

    normalize = mcolors.Normalize(vmin=np.min(Xs), vmax=np.max(Xs))
    fig, ax = plt.subplots(figsize=figsize)
    for i, x in enumerate(Xs):
        y, u = generate_recur_2d(recur, x, max_rate=5.)
        color = colormap(normalize(x))
        ax.scatter(y, u, marker='.', color=color, label=ylabel_fm.format(x))
        ax.set_xlabel(xlabel, fontdict=font)
        ax.set_ylabel(r"$\lim_{n \rightarrow \infty} u_n$", fontdict=font)
        ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
        ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.legend(loc='upper right')
        # save it
    save_path = "../figure/bifurcation/{}".format(name)
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
    print("Save file: {}.png".format(save_path))


def plot_chaotic_rq(name, figsize=(5, 5), xlabel=r"$\frac{1}{\ell^2}$", yticks=None):
    # picking sigma2 = 0.1,0.5, 1.,2.,4., 6.
    sigma2s = np.array([0.33, 0.5, 1., 2.])
    alphas = np.array([0.5, 1., 3.])

    # choose color map
    colormap = cm.viridis

    normalize = mcolors.Normalize(vmin=np.min(sigma2s[0] * alphas[0]), vmax=np.max(sigma2s[-1] * alphas[-1]))
    fig, ax = plt.subplots(figsize=figsize)
    for i, sigma2 in enumerate(sigma2s):
        for alpha in alphas:
            y, u = generate_recur_rq(sigma2, alpha, max_rate=5.)
            color = colormap(normalize(alpha * sigma2))
            ax.scatter(y, u, marker='.', color=color, label=r'$\sigma^2={}, \alpha={}$'.format(sigma2, alpha))
            ax.set_xlabel(xlabel, fontdict=font)
            ax.set_ylabel(r"$\lim_{n \rightarrow \infty}u_n$", fontdict=font)
            ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
            ax.set_xticks([0.5, 1, 2.5, 3])
            if yticks is not None:
                ax.set_yticks(yticks)
            ax.legend()
    # save it
    save_path = "../figure/bifurcation/{}".format(name)
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
    print("Save file: {}.png".format(save_path))


def plot_chaotic_period(name, lengthscale, figsize=(5, 5), xlabel=r"$\frac{1}{p^2}$", yticks=None, xticks=None):
    sigma2s = np.array([0.33, 0.5, 1., 2.])
    colormap = cm.viridis
    normalize = mcolors.Normalize(vmin=sigma2s[0], vmax=sigma2s[-1])
    fig, ax = plt.subplots(figsize=figsize)
    for i, sigma2 in enumerate(sigma2s):
        inv_lengthscale = 1. / lengthscale ** 2
        y, u = generate_recur_per(sigma2, inv_lengthscale, max_rate=5.)
        color = colormap(normalize(sigma2))
        ax.scatter(y, u, marker='.', color=color, label=r'$\sigma^2={}$'.format(sigma2))
        ax.set_xlabel(xlabel, fontdict=font)
        ax.set_ylabel(r"$\lim_{n \rightarrow \infty}u_n$", fontdict=font)
        ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
        if xticks is not None:
            ax.set_xticks(xticks)

        if yticks is not None:
            ax.set_yticks(yticks)
        ax.legend(loc='upper right',borderpad=0.8)
        ax.set_title(r'$\ell={}$'.format(lengthscale))
        # save it
    save_path = "../figure/bifurcation/{}_ell_{}".format(name, lengthscale)
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
    print("Save file: {}.png".format(save_path))

def plot_chaotic_period_fix_sigma(name,
                                  sigma2,
                                  figsize=(5, 5),
                                  xlabel=r"$\frac{1}{p^2}$",
                                  yticks=None,
                                  xticks=None ):
    lengthscales = np.array([0.8, 1., 2., 4.])
    colormap = cm.viridis
    normalize = mcolors.Normalize(vmin=lengthscales[0], vmax=lengthscales[-1])
    fig, ax = plt.subplots(figsize=figsize)
    for lengthscale in lengthscales:
        inv_lengthscale = 1. / lengthscale ** 2
        y, u = generate_recur_per(sigma2, inv_lengthscale, max_rate=5.)
        color = colormap(normalize(lengthscale))
        ax.scatter(y, u, marker='.', color=color, label=r'$\ell={}$'.format(lengthscale))
        ax.set_xlabel(xlabel, fontdict=font)
        ax.set_ylabel(r"$\lim_{n \rightarrow \infty}u_n$", fontdict=font)
        ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.legend(loc='upper right',borderpad=0.8)
        ax.set_title(r'$\sigma^2={}$'.format(sigma2))

    save_path = "../figure/bifurcation/{}_sigma2_{}".format(name, sigma2)
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
    print("Save file: {}.png".format(save_path))

def contour(name,
            recur=recur_se,
            figsize=(5, 5),
            xlabel=r"$\sigma^2$",
            ylabel=r"$\ell$",
            min_x=0.,
            min_y=0.,
            max_x=10.,
            max_y=10.,
            num_x=100,
            num_y=100,
            title=""):

    file_name = name + ".pkl"
    save_file = os.path.join("../data/bifurcation", file_name)
    if os.path.exists(save_file):
        print("Saved data exists! Load {}".format(save_file))
        with open(save_file, 'rb') as f:
            Us = pickle.load(f)
    else:
        X_Y, Us = generate_recur_3d(recur=recur,
                                    max_y=max_y,
                                    max_x=max_x,
                                    min_x= min_x,
                                    min_y= min_y,
                                    num_x=num_x,
                                    num_y=num_y,
                                    num_discard=300,
                                    num_gen=1)
        with open(save_file, 'wb') as f:
            pickle.dump(Us, f)
            print("Write {}!".format(save_file))
    fig, ax = plt.subplots(figsize=figsize)
    X, Y = np.meshgrid(np.linspace(min_x, max_x, num_x), np.linspace(min_y, max_y, num_y))
    CS = ax.contour(X, Y, np.array(Us).reshape(num_x, num_y))
    ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(xlabel, fontdict=font)
    ax.set_ylabel(ylabel, fontdict=font)
    ax.set_title(title, fontdict=font)
    save_path = "../figure/bifurcation/{}_contour".format(name)
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
    print("Save file: {}.png".format(save_path))


## SE (Figure 4 a b)
# contour(name='se_3d', recur=recur_se)
# plot_chaotic(name='se', recur=recur_se)

## SE HIGH DIMENSIONAL (Figure 12)
# m >=2
# m = 2
# recure_se_high_dim_v3 = lambda x, sigma2, rate : recure_se_high_dim(x, sigma2, rate, m)
# plot_chaotic(name="se_m_{}_v3".format(m), recur=recure_se_high_dim_v3, xticks=[0.5, 2, 4])
# contour(name='se_3d_m_{}_v3'.format(m), recur=recure_se_high_dim_v3)
#
# m = 3
#recure_se_high_dim_v3 = lambda x, sigma2, rate : recure_se_high_dim(x, sigma2, rate, m)
# plot_chaotic(name="se_m_{}_v3".format(m), recur=recure_se_high_dim_v3, xticks=[0.5, 2, 4])
# contour(name='se_3d_m_{}_v3'.format(m), recur=recure_se_high_dim_v3)

# m = 4
# plot_chaotic(name="se_m_{}_v3".format(m), recur=recure_se_high_dim_v3, xticks=[0.5, 2, 4])
# contour(name='se_3d_m_{}_v3'.format(m), recur=recure_se_high_dim_v3)
#
# m = 5
# plot_chaotic(name="se_m_{}_v3".format(m), recur=recure_se_high_dim_v3, xticks=[0.5, 2, 4])
# contour(name='se_3d_m_{}_v3'.format(m), recur=recure_se_high_dim_v3)
#
# m = 6
# plot_chaotic(name="se_m_{}_v3".format(m), recur=recure_se_high_dim_v3, xticks=[0.5, 2, 4])
# contour(name='se_3d_m_{}_v3'.format(m), recur=recure_se_high_dim_v3)

## INPUT-CONNECTED (Figure 11)
# plot_chaotic(name="input_connected", recur=recur_input_connected, yticks=[0,0.5, 2, 4, 6, 8, 10])

## COSIN (Figure 5a)
# plot_chaotic(name='cosin', recur=recur_cosin, xlabel=r'$\pi^2/p^2$', xticks=[0.5, 1, 2, 3])
# contour(name='cosine_3d', recur=recur_cosin, ylabel=r'$\pi^2/p^2$')

## PERIODIC (Figure 13)
# lengthscale = 0.8
# plot_chaotic_period(name="periodic", lengthscale=lengthscale, xticks=[0, 0.5, 1, 2, 4])
#
# lengthscale = 1.
# plot_chaotic_period(name="periodic", lengthscale=lengthscale, xticks=[0, 0.5, 1, 2, 4])
#
# lengthscale = 2.
# plot_chaotic_period(name="periodic", lengthscale=lengthscale, yticks=[0, 0.4, 0.8], xticks=[0, 0.5, 1, 2, 4])
#
# lengthscale = 4.
# plot_chaotic_period(name="periodic", lengthscale=lengthscale, yticks=[0, 0.15, 0.25], xticks=[0, 0.5, 1, 2, 4])
#
# sigma2 = 0.5
# plot_chaotic_period_fix_sigma("period", sigma2=sigma2, xticks=[0, 0.5, 1, 2, 4])
#
# sigma2 = 1
# plot_chaotic_period_fix_sigma("period", sigma2=sigma2,
#                               # xticks=[0, 1, 2, 4],
#                               # yticks=[0, 0.2, 0.4]
#                               )
#
# sigma2 = 2.
# plot_chaotic_period_fix_sigma("period", sigma2=sigma2,
#                               # xticks=[0, 1, 2, 4],
#                               # yticks=[0, 0.4, 0.8]
#                               )
#
# sigma2 = 4.
# plot_chaotic_period_fix_sigma("period", sigma2=sigma2,
#                               # xticks=[0, 1, 2, 4],
#                               # yticks=[0, 0.8, 1.5]
#                               )

## PERIODIC CONTOUR (Figure 13)
## note that recur_periodic_countour takes 1/p^2 and \ell as inputs
# contour(name="per_contour", recur=recur_periodic_contour,
#         min_x=0.1,
#         max_x=3,
#         min_y=0.5,
#         max_y=5.,
#         xlabel=r'$1/p^2$',
#         ylabel=r'$\ell$')


# plot_chaotic(name="periodic", recur=recur_periodic, xlabel=r'$p$')

#### RATIONAL QUADRATIC (Figure 14)
# plot_chaotic_rq(name="rq_sigma_alpha")
## contour plots
# alpha=0.5
# contour(name="rq_alpha_{}".format(alpha),
#         recur=recur_rational_quadratic_contour,
#         min_x=0.1,
#         max_x=5,
#         min_y=0.1,
#         max_y=5.,
#         num_x=50,
#         num_y=50,
#         xlabel=r'$1/\ell^2$',
#         ylabel=r'$\sigma^2$',
#         title=r'$\alpha={}$'.format(alpha))

# alpha=3.
# contour(name="rq_alpha_{}".format(alpha),
#         recur=recur_rational_quadratic_contour,
#         min_x=0.1,
#         max_x=5,
#         min_y=0.1,
#         max_y=5.,
#         num_x=50,
#         num_y=50,
#         xlabel=r'$1/\ell^2$',
#         ylabel=r'$\sigma^2$',
#         title=r'$\alpha={}$'.format(alpha))


### SM (Figure 5d)
contour(name='SM_1d',
        recur=recur_sm_1d,
        min_x=0.01,
        max_x=1.,
        min_y=0.01,
        max_y=1.,
        num_x=50,
        num_y=50,
        xlabel=r'$\sigma^2$',
        ylabel=r'$\mu$'
        )
#
# plot_chaotic("SM_1d",
#              recur=recur_sm_1d
#              )


plt.show()
