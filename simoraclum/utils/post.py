import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def target_parity(config, data):
    target_df = pd.DataFrame()
    target_df[config.target] = data["energies"]
    for col in list(data.columns):
        try:
            if col.split("_")[0] == config.target:
                target_df[col] = data[col]
        except IndexError:
            continue

    return target_df


def plot_candidates(target, df, name="tmp"):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title(f"Individual Relaxed Samples and their {target}")
    data = df.values
    l = data.shape[-1]
    ax.plot(data[:, 0], data[:, 0], "--")
    for s, sample in enumerate(data):
        color_scale = [abs(data[s, 0] - i) for i in sample]
        ax.scatter([data[s, 0]] * l, sample, c=color_scale)
    if l > 2:
        ax = fig.add_subplot(2, 1, 2, sharex=ax)
        ax.set_title(f"Mean of relaxed structures for each sample")
        ax.plot(data[:, 0], data[:, 0], "--")
        x = data[:, 0]
        ym = np.mean(data[:, 1:], axis=1)
        ys = np.std(data[:, 1:], axis=1)
        ax.scatter(x, x)
        ax.scatter(x, ym)
        ax.errorbar(x, ym, yerr=ys, fmt="none")
    fig.supxlabel(f"True {target}")
    fig.supylabel("Relaxed Target predicted with chosen oracle")
    fig.tight_layout()
    fig.savefig(name)
