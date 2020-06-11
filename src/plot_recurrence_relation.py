"""This code is to plot the shape of recurrence relations
Plots in the paper:
    - Figure 6 (first plot)
    - Figure 15
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mpmath import hyp2f0, hyp1f1

# plot setup
matplotlib.rcParams.update({
    'font.size': 18,
    'figure.subplot.bottom': 0.125,
})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 36,
        }


def recur_rational_quadratic(x, alpha, m):
    """

    :param x:
    :param alpha: \alpha
    :param rate: 1/\ell^2
    :return:
    """
    z = -  x / (alpha)
    fz = hyp2f0(alpha, m / 2, z)
    return 2 * (1. - fz.real)


def se_1d(x):
    return 2 * (1 - 1. / np.sqrt(1 + x))


def se_high_dim(x, m):
    return 2 * (1. - 1. / np.power(1 + x, 0.5 * m))


def spectral_mixture(x, m):
    half_m = m / 2.
    return 2 * (1 - np.exp(-m * x / (1 + x)) / np.power(1 + x, half_m))


def cosin_1d(x):
    return 2 * (1 - np.exp(-x))


def recur_cosin_high_dim(x, m):
    fz = hyp1f1(m / 2., 1. / 2., -x)

    if fz.imag > 0:
        print("hyp1f1 has imaginary part")
    return 2 * (1 - fz.real)


def recur_periodic(x, inv_lengthscale2):
    return 2 * inv_lengthscale2 * (1 - np.exp(-np.pi ** 2 * x))


plt.subplots(figsize=(6, 6))
lw = 4
x = np.linspace(0, 2, 50)
t = 0.8


#


def plot_se_only():
    m = 1
    plt.plot(x, se_high_dim(x, m), '-.', lw=lw, label=r'SE, $m={}$'.format(m))
    # ## odd case
    m = 4
    plt.plot(x, se_high_dim(x, m), '-.', lw=lw, label=r'SE, $m={}$'.format(m))

    m = 10
    plt.plot(x, se_high_dim(x, m), '-.', lw=lw, label=r'SE, $m={}$'.format(m))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.xlim([0, 2])
    plt.ylim([0, 2])

    plt.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    plt.xlabel(r'$x$', fontdict=font)
    plt.ylabel(r'$h(x)$', fontdict=font)
    plt.savefig("../figure/bifurcation/h_x_se.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_rq_only():
    m = 1
    alpha = 0.3
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    alpha = 1.
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    m = 3
    alpha = 0.3
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    m = 1
    alpha = 1.
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    m = 6
    alpha = 0.3
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))
    #
    alpha = 1.
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.xlim([0, 2])
    plt.ylim([0, 2])

    plt.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    plt.xlabel(r'$x$', fontdict=font)
    plt.ylabel(r'$h(x)$', fontdict=font)
    plt.savefig("../figure/bifurcation/h_x_rq.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_sm_only():
    m = 1
    plt.plot(x, spectral_mixture(x, m), 'o', lw=lw, label=r'SM $m={}$'.format(m))

    m = 4
    plt.plot(x, spectral_mixture(x, m), 'o', lw=lw, label=r'SM $m={}$'.format(m))

    m = 10
    plt.plot(x, spectral_mixture(x, m), 'o', lw=lw, label=r'SM $m={}$'.format(m))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.xlim([0, 2])
    plt.ylim([0, 2])

    plt.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    plt.xlabel(r'$x$', fontdict=font)
    plt.ylabel(r'$h(x)$', fontdict=font)
    plt.savefig("../figure/bifurcation/h_x_sm.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_mix():
    # SE
    m = 1
    plt.plot(x, se_high_dim(x, m), '-.', lw=lw, label=r'SE, $m={}$'.format(m))
    # ## odd case
    m = 4
    plt.plot(x, se_high_dim(x, m), '-.', lw=lw, label=r'SE, $m={}$'.format(m))

    # RQ
    m = 1
    alpha = 1.
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    m = 1
    alpha = 0.3
    y = [recur_rational_quadratic(x_i, alpha, m) for x_i in x]
    plt.plot(x, y, 'd', lw=lw, label=r'RQ, $\alpha={}, m={}$'.format(alpha, m))

    # COS\
    plt.plot(x, cosin_1d(x), 'x', lw=lw, label=r'COS')

    # SM
    m = 1
    plt.plot(x, spectral_mixture(x, m), 'o', lw=lw, label=r'SM $m={}$'.format(m))

    m = 4
    plt.plot(x, spectral_mixture(x, m), 'o', lw=lw, label=r'SM $m={}$'.format(m))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.xlim([0, 2])
    plt.ylim([0, 2])

    plt.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    plt.xlabel(r'$x$', fontdict=font)
    plt.ylabel(r'$h(x)$', fontdict=font)
    plt.savefig("../figure/bifurcation/h_x.png", bbox_inches='tight', dpi=300)
    plt.show()


## Figure 16
# plot_sm_only()

# plot_se_only()

plot_rq_only()

## Figure 6
# plot_mix()
