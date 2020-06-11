"""This code is to plot the bifurcation plot for the logistic map
Plot in the paper:
 - Figure 2
"""

import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

# plot setup
matplotlib.rcParams.update({
    'font.size': 24,
    'figure.subplot.bottom': 0.125,
})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 36,
        }


## Define recurrence relation

def recur_se(x, rate):
    """Recurrence relation for SE kernel"""
    sigma2 = 1.
    return sigma2 * (1. - 1. / np.sqrt(1 + rate * x))


def recur_chaos(x, rate):
    """Dynamic system in the background section"""
    return rate * x * (1. - x)


def generate_recur(recur, num_discard=500, num_gen=200, min_rate=0., max_rate=5.0, num_rate=500):
    rates = np.linspace(min_rate, max_rate, num_rate)
    x_inital = 0.1
    X, Y = [], []

    # burn-in phase
    for rate in rates:
        x = x_inital
        for _ in range(num_discard):
            z = recur(x, rate)
            x = z

        for _ in range(num_gen):
            z = recur(x, rate)
            x = z
            X.append(rate)
            Y.append(z)

    return X, Y


def plot_chaotic(recur, fig_name, figsize=(10, 6), max_rate=5., xticks=[0, 1, 2, 3, 4], ylabel=r"$u_n$"):
    X, Y = generate_recur(recur, max_rate=max_rate)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X, Y, ls='', marker='.', markersize='0.5')
    ax.set_xlabel(r"$r$", fontdict=font)
    ax.set_ylabel(ylabel, fontdict=font)
    ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    ax.set_xticks(xticks)
    ax.set_yticks([0, 0.5, 1])
    # if xtick_labels is not None:
    #     ax.set_xticklabels(xtick_labels, fontdict=font)
    # save it
    save_path = "../figure/bifurcation/{}".format(fig_name)
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)


def arrowed_spines(fig, ax):
    """Making arrow for spines"""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)

    plt.xticks([])
    plt.yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    hw = 1. / 20. * (ymax - ymin)
    hl = 1. / 20. * (xmax - xmin)
    lw = 1.  # axis line width
    ohg = 0.01  # arrow overhang

    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
             head_width=hw, head_length=hl, overhang=ohg,
             length_includes_head=True, clip_on=False)

    ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
             head_width=yhw, head_length=yhl, overhang=ohg,
             length_includes_head=True, clip_on=False)


def plot_chaotic_2(recur, fig_name, figsize=(7, 7), max_rate=5., xticks=[0, 1, 2, 3, 4]):
    X, Y = generate_recur(recur, max_rate=max_rate)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X, Y, ls='', marker='.', markersize='2.5')
    ax.set_xlabel(r"$\theta$", fontdict=font)
    ax.set_ylabel(r"$\lim_{l\rightarrow \infty}{E}[Z_l]$", fontdict=font)
    ax.set_xticks([])
    ax.set_yticks([])

    arrowed_spines(fig, ax)
    # save it
    save_path = "../figure/bifurcation/{}".format(fig_name)
    plt.tight_layout()
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)


# plot_chaotic_2(recur=recur_chaos, fig_name='chaotic_2')


plot_chaotic(recur=recur_chaos,
             figsize=(7, 5),
             fig_name='chaotic',
             xticks=[0, 1, 2, 3.45, 4],
             ylabel=r'$\lim_{l \rightarrow \infty} u_l$')

plt.show()
