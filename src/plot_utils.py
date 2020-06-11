import os
import numpy as np
import matplotlib
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from matplotlib import rc, lines as mlines
import matplotlib.patches as mpatches
import pickle
import seaborn as sns

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

def plot_dual_ez_rmse_by_dimension(save_dir, n_layers, save_file, selected_dim=2):

    # cmap = matplotlib.cm.get_cmap('Spectral')
    # norm = matplotlib.colors.Normalize(vmin=n_layers[0], vmax=n_layers[-1])
    palette = sns.color_palette("colorblind", len(n_layers))

    rmses = []
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()
    for i, n_layer in enumerate(n_layers):


        file_name = "dim_{}_layer{}".format(selected_dim, n_layer)
        with open(os.path.join(save_dir, file_name + "_rmse.pkl"), 'rb') as f:
            RMSEs = pickle.load(f)
        rmses.append(np.array(RMSEs).mean())

        with open(os.path.join(save_dir, file_name + "_EZ.pkl"), 'rb') as f:
            EZ = pickle.load(f)
            EZ = EZ.squeeze().tolist()

        mean_RMSE = np.array(RMSEs).mean()
        last_EZ = EZ[-1]
        con = ConnectionPatch(xyA=(n_layer, last_EZ ), xyB=(n_layer, mean_RMSE), coordsA="data", coordsB="data",
                              axesA=ax1,
                              axesB=ax2,
                              shrinkA=8,
                              shrinkB=8,
                              arrowstyle="-|>",
                              linestyle=(0, (5, 10)),
                              color=palette[i])
        ax1.add_artist(con)

        ax1.plot(list(range(1, len(EZ) + 1)), EZ, '-', marker='o', linewidth=2.5, markersize=8, color=palette[i], label=r"$L={}$".format(n_layer))
        ax2.plot(n_layer, np.array(RMSEs).mean(), marker='*', color=palette[i], markersize=18, label=r"$L={}$".format(n_layer))


        # ax2.set_ylim([1.5, 3.8])
        ax1.set_xlabel("Layer", fontdict=font)
        ax1.set_ylabel(r"$E[Z]/\sigma^2$", fontdict=font)
        ax2.set_ylabel("RMSE", fontdict=font)

    ax1.set_title(r"$m={}$".format(selected_dim), fontdict=font)
    # ax1.legend()

    fig.savefig(save_file, bbox_inches='tight', dpi=300)


def get_legend(n_layers):

    fig = plt.figure(figsize=(1, 1.25))
    labels = []
    colors = []
    palette = sns.color_palette("colorblind", len(n_layers))
    for i, n_layer in enumerate(n_layers):
        labels += [r"$N={}$".format(n_layer)]
        colors += [palette[i]]

    patches = [
        mlines.Line2D([], [],
                      color=color,
                      label = label,
                      linestyle="-", marker='o',markersize=8, linewidth=2.5)
        for label, color in zip(labels, colors)]
    fig.legend(patches, labels, loc='center', frameon=False)
    fig.savefig("legend.png", dpi=300)
