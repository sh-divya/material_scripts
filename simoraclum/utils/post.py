import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator


def target_parity(target, data):
    target_df = pd.DataFrame()
    target_df[target] = data[target]
    for col in list(data.columns):
        try:
            if col.split("_")[0] == target:
                target_df[col] = data[col]
        except IndexError:
            continue

    return target_df


def parse_times(df):
    pass


def parse_structs(samples, minidx):
    cols = samples.columns
    struct_df = pd.DataFrame()
    tmp = []
    for c, col in minidx.items():
        name = "Struct_" + col.split("_")[-1]
        tmp.append(samples.loc[c, name])
    struct_df["Best"] = tmp
    try:
        struct_df["cif"] = samples["cif"]
    except IndexError:
        pass
    struct_df = struct_df.dropna()

    struct_df = struct_df.map(lambda x: Structure.from_str(x, fmt="cif"))
    return struct_df


def target_delta(res):
    true = res.iloc[:, 0]
    pred = res.iloc[:, 1:]
    delta = abs(pred.sub(true, axis=0))
    df = pd.DataFrame()
    df["Best"] = delta.min(axis=1)
    df["Mean"] = delta.mean(axis=1)
    df["iBest"] = delta.idxmin(axis=1)
    return df


def struct_dist(structs):
    pass


def single_rmsd(str1, str2):
    matcher = StructureMatcher(
        primitive_cell=False,
        ltol=4.0,
        stol=4.0,
        angle_tol=15,
        comparator=ElementComparator(),
    )
    d, maxd = matcher.get_rms_dist(str1, str2)
    return d, maxd


def struct_rmsd(structs):
    rmsd = pd.DataFrame()
    true = structs["cif"]
    pred = structs["Best"]
    d = []
    for s, struct in pred.items():
        d.append(single_rmsd(struct, true.iloc[s]))
    d1, d2 = list(zip(*d))
    rmsd["RMSD"] = d1
    rmsd["maxD"] = d2
    return rmsd


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
    fig.supylabel(f"Relaxed {target} predicted with chosen oracle")
    fig.tight_layout()
    return fig
