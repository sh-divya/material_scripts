import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

BASE_PATH = Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / "results"
PLOTS_PATH = BASE_PATH / "plots"

if __name__ == "__main__":
    results_fptr = [
        "filter_ngen5_steps500_m3gnet_distro.csv",
        "train_ngen5_steps500_m3gnet_distro.csv"
    ]

    names = ["CGFN-Relaxed", "Matbench"]
    target = "Eform"
    nsamples = 1000
    t = pd.DataFrame()
    d1 = []
    d2 = []
    sns.set_theme()
    fig = plt.figure()
    ax = fig.gca()

    for f, fptr in enumerate(results_fptr):
        res_path = RESULTS_PATH / fptr
        data = pd.read_csv(res_path)
        t[names[f]] = data[target]
        d1.append(data["MinD"])
        d2.append(data["MaxD"])
    w1 = wasserstein_distance(t[names[0]], t[names[1]])
    w2 = wasserstein_distance(*d1)
    w3 = wasserstein_distance(*d2)
    print(w1, w2, w3)
    ax = sns.kdeplot(data=t, multiple="stack", alpha=0.5)
    ax.set_xlabel(f"{target}(eV)")
    ax.set_title(f"Kernel Density Estimation of {target} distribution")
    fig.savefig(PLOTS_PATH / "target_kde.png")