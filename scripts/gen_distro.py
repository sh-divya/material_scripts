import pandas as pd
from pathlib import Path
from src.utils.io import get_saved_results
from src.utils.post import minmaxD
from matplotlib import pyplot as plt
import click
import numpy as np

BASE_PATH = Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / "results"
PLOTS_PATH = BASE_PATH / "plots"

@click.command()
@click.option("--csv", default="train_data1000_ngen5_steps500_m3gnet_verbose.csv")
@click.option("--target", default="Eform")
def gen(csv, target):
    gen_base(csv, target)

def gen_base(csv, target="Eform"):
    distro_df = pd.DataFrame()
    fptr = RESULTS_PATH / csv
    fname = fptr.stem
    fname = fname.split("_")
    sim_type = fname[0]
    data = pd.read_csv(fptr)

    if sim_type == "train":
        print(fname)
        samples = int(fname[1][4:])
        ngen = int(fname[2][4:])
        iter_steps = int(fname[3][5:])
        fname = [fname[0]] + fname[2:5] + ["distro.csv"]
        distro_df[target] = data[target]
        distro_df["cif"] = data["cif"]
    else:
        samples = fname[2][7:]
        ngen = int(fname[3][4:])
        iter_steps = int(fname[4][5:])
        rel_structs, target_vals, _ = get_saved_results(fptr, target, ngen, samples)
        imin = np.array(target_vals.argmax(axis=1))
        min_val = target_vals[np.array(list(range(int(samples)))), imin]
        min_struct = rel_structs[np.array(list(range(int(samples)))), imin]
        distro_df[target] = min_val
        distro_df["cif"] = min_struct
        distro_df = distro_df.dropna()
        fname = [fname[0]] + fname[3:6] + ["distro.csv"]
    # distro_df = distro_df.iloc[:5, :]
    mind, maxd = minmaxD(distro_df)
    distro_df["MinD"] = mind
    distro_df["MaxD"] = maxd
    distro_df = distro_df.drop(columns=["cif"], axis=1)
    distro_df.to_csv(RESULTS_PATH / "_".join(fname))



if __name__ == "__main__":
    gen_base("train_data1000_ngen5_steps500_m3gnet_verbose.csv")
    # gen_base("filter_gfn_samples1000_ngen5_steps500_m3gnet_verbose.csv")    