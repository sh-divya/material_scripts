import pandas as pd
from pathlib import Path
from simoraclum.utils.post import (
    target_parity,
    plot_candidates,
    target_delta,
    parse_structs,
    struct_rmsd,
)
from argparse import ArgumentParser
from matplotlib import pyplot as plt

BASE_PATH = Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / "results"
PLOTS_PATH = BASE_PATH / "plots"

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--type", default="train")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--target", default="Eform")
    args = parser.parse_args()

    sim_type = args.type
    separate_time = args.time
    target = args.target

    for fptr in RESULTS_PATH.iterdir():
        fname = fptr.stem
        if fptr.suffix == ".csv":
            fname = fname.split("_")
            if fname[0] == sim_type:
                data = pd.read_csv(fptr)
                if sim_type == "train":
                    print(fname)
                    samples = int(fname[1][4:])
                    ngen = int(fname[2][4:])
                    iter_steps = int(fname[3][5:])
                    if separate_time:
                        tptr = f"times{samples}_ngen{ngen}"
                        times = pd.read_csv(
                            RESULTS_PATH / tptr, names=[str(i) for i in range(ngen)]
                        )
                        times[times.columns[0]] = times[times.columns[0]].str[1:]
                        times[times.columns[-1]] = times[times.columns[-1]].str[:-1]
                        times = times.astype("float")
                    else:
                        # get times
                        pass
                else:
                    samples = fname[2][7:]
                    ngen = int(fname[3][4:])
                    iter_steps = int(fname[4][5:])
                data = data.iloc[:5, :]
                res = target_parity(target, data)
                err = target_delta(res)
                struct_df = parse_structs(data, err["iBest"])
                err = err.drop(columns=["iBest"], axis=1)
                res = res.apply(lambda x: pd.to_numeric(x, errors="coerce")).dropna()
                err[["RMSD", "maxD"]] = struct_rmsd(struct_df)
                fig = plot_candidates(
                    target, res, PLOTS_PATH / ("_".join(fname) + ".png")
                )
                fig, ax = plt.subplots()
                err.hist(ax=ax)
                fig.savefig(PLOTS_PATH / ("_".join(fname) + "_performance.png"))
                print(err.describe())
                raise Exception
