import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mpmath import hyp2f0, nstr

# plot setup
matplotlib.rcParams.update({
    'font.size': 24,
    'figure.subplot.bottom': 0.125,
})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 24,
        }


def spectral_mixture(x, m=4):
    half_m = m / 2.
    return 2 * (1 - np.exp(-m * x / (1 + x)) / np.power(1 + x, half_m))


def recur_rational_quadratic(x, alpha=0.5, m=1):
    y = []
    try:
        for x_i in x:
            z = -  x_i / (alpha)
            fz = hyp2f0(alpha, m / 2, z)
            ret = 2 * (1. - fz.real)
            ret = nstr(ret)
            y.append(float(ret))
    except:
        z = -  x / (alpha)
        fz = hyp2f0(alpha, m / 2, z)
        ret = 2 * (1. - fz.real)
        ret = nstr(ret)
        return float(ret)
    return np.array(y)


def plot_system(recur, x0, n, title, name, xticks):
    """Plot the path of iterations which converges to a fixed point"""
    fig, ax = plt.subplots(figsize=(5, 5))
    t = np.linspace(0, 2)
    ax.plot(t, recur(t), 'r', lw=4, label=r'$y = h(x)$')
    ax.plot([0, 2], [0, 2], 'g', lw=4, label=r'$y = x$')
    x = x0
    for i in range(n):
        y = recur(x)
        ax.plot([x, x], [x, y], 'b', lw=2)
        ax.plot([x, y], [y, y], 'b', lw=2)
        ax.plot([x], [y], 'ok', ms=8)
        x = y

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks(xticks)
    ax.set_yticks([0, 1, 2])
    ax.grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)
    ax.set_title(title)
    ax.legend()
    fig.savefig("../figure/fixed_points/{}.png".format(name), bbox_inches='tight', dpi=300)


plot_system(spectral_mixture, 0.6, 5, r'SM $m=4$', 'fp_sm_4d', xticks=[0, 0.6, 1, 2])

plot_system(recur_rational_quadratic, 1.2, 5, r'RQ $\alpha=0.5$', 'fp_rq_0_5', xticks=[0, 1, 1.2, 2])

plt.show()
