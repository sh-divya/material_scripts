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
RESULTS_PATH = BASE_PATH / "results2"
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
            if fname[-1] != "results":
                if fname[0] == sim_type:
                    data = pd.read_csv(fptr)
                    if sim_type == "train":
                        print(fname)
                        samples = int(fname[1][4:])
                        ngen = int(fname[2][4:])
                        iter_steps = int(fname[3][5:])
                        fname = [fname[0]] + fname[2:5] + ["results.csv"]
                        if separate_time:
                            tptr = f"times{samples}_ngen{ngen}.txt"
                            times = pd.read_csv(
                                RESULTS_PATH / tptr, names=[str(i) for i in range(ngen)]
                            )
                            times[times.columns[0]] = times[times.columns[0]].str[1:]
                            times[times.columns[-1]] = times[times.columns[-1]].str[:-1]
                            times = times.astype("float")
                        else:
                            times = pd.DataFrame()
                            pass
                    else:
                        samples = fname[2][7:]
                        ngen = int(fname[3][4:])
                        iter_steps = int(fname[4][5:])
                        fname = [fname[0]] + fname[3:6] + ["results.csv"]
                    data = data.iloc[:5, :]
                    try:
                        times = times.iloc[:5, :]
                    except IndexError:
                        pass
                    pd.set_option('display.max_rows', None)
                    res = target_parity(target, data)
                    res = res.apply(lambda x: pd.to_numeric(x, errors="coerce"))
                    err = target_delta(res)
                    err = err.dropna()
                    struct_df = parse_structs(data, err["iBest"])
                    err = err.drop(columns=["iBest"], axis=1)
                    err[["Best RMSD", "Best maxD"]] = struct_rmsd(struct_df)
                    struct_df = parse_structs(data, err["iChoice"])
                    err = err.drop(columns=["iChoice"], axis=1)
                    err[["RMSD with min", "maxD with min"]] = struct_rmsd(struct_df)
                    err = pd.concat([err, times], axis=1)
                    print(fname)
                    print(err)
                    err.to_csv(RESULTS_PATH / "_".join(fname))
                    fig = plot_candidates(
                        target, res, PLOTS_PATH / ("_".join(fname) + ".png")
                    )
