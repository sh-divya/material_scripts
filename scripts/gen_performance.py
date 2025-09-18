import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from matplotlib import pyplot as plt

BASE_PATH = Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / "results2"
PLOTS_PATH = BASE_PATH / "plots"

if __name__ == "__main__":

    filtered_gen = "filter_ngen5_steps500_m3gnet_results.csv"
    sorted_gen = "sorted_ngen5_steps500_m3gnet_results.csv"
    reference_data = "train_ngen5_steps500_m3gnet_results.csv"
    nsamples = 100

    best_cols = ["Best", "Mean", "Best RMSD", "Best maxD"]
    min_cols = ["Choice", "RMSD with min", "maxD with min"]

    for gen in [filtered_gen, sorted_gen]:
        res_path = RESULTS_PATH / gen
        ref_path = RESULTS_PATH / reference_data
        fname = gen.split(".")[0].split("_")
        ngen = fname[1][4:]
        iters = fname[2][5:]
        time_cols = [str(i) for i in range(ngen)]

        err_rmsd_times = pd.read_csv(res_path)
        best_data = err_rmsd_times[best_cols]
        min_data = err_rmsd_times[min_cols]
        time_data = err_rmsd_times[time_cols]


        fig1, ax1 = plt.subplots()
        best_data.hist(ax=ax1)
        fig2, ax2 = plt.subplots()
        min_data.hist(ax=ax2)

        # fig3, ax3 = plt.subplots()
        # time_data.hist(ax=ax3)
        # fig.savefig(PLOTS_PATH / ("_".join(fname) + "_performance.png"))
        print(best_data.describe())
        print(min_data.describe())
        # print("RMSD nans", samples - err["RMSD"].shape[0])
        # print("maxD nans", samples - err["maxD"].shape[0])