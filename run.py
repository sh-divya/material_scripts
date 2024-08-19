from simoraclum.utils.io import (
    sample_candidates,
    struct_generator,
    save_results,
    get_saved_results,
)
from simoraclum.utils.post import target_parity, plot_candidates
from simoraclum.oracles.mlff import MLOracle

# from pymatgen.core import Structure
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from yaml import safe_load
import numpy as np

BASE_PATH = Path(__file__).parent
CONFIG_PATH = BASE_PATH / "config"


def get_relaxed_structures(oracle, structs, verbosity):
    rel_str = []
    for s in structs:
        if s is not None:
            rel_str.append(oracle.relax(s, verbosity))
        else:
            rel_str.append(None)
    return rel_str


def get_delta_target(oracle, structs, true_val):
    pred_vals = []
    tmp_idx = []
    for i, s in enumerate(structs):
        if s is not None:
            pred_vals.append(oracle.predict(s).item())
        else:
            pred_vals.append(10000)
            tmp_idx.append(i)
    delta = [(i, abs(val - true_val)) for i, val in enumerate(pred_vals)]
    for i in tmp_idx:
        pred_vals[i] = None
    return delta, pred_vals


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="mlff")
def relax(cfg: DictConfig):
    data_file, num, rnd, start_idx = (
        cfg.data_file,
        cfg.nsamples,
        cfg.random,
        cfg.start,
    )
    ngen, verbose, target, wyck = (cfg.ngen, cfg.verbose, cfg.target, cfg.wyckoff)
    samples = sample_candidates(data_file, num, rnd)
    # samples = pd.read_csv(BASE_PATH / "data" / data_file, index_col=0)
    fptr = data_file.split(".")[0]
    v = "_verbose" if verbose else ""
    w = "_wyck" if wyck else ""
    fptr = f"{fptr}{num}_ngen{ngen}_steps{cfg.rel_iter}_{cfg.mlff}{w}{v}.csv"
    # fptr = f"filter_gfn_samples10_ngen5_steps250_m3gnet_verbose.csv"
    rel_structs, rel_targets, pyx_times = get_saved_results(fptr, target, ngen, num)
    oracle = MLOracle(cfg.mlff, cfg.target, cfg.rel_iter)
    if wyck:
        wyck_dic = safe_load(open(cfg.wyckoff))
    else:
        wyck_dic = False
    for i, sample in tqdm(enumerate(samples.iterrows())):
        if i < start_idx:
            print("Skipping", i)
            continue
        s, sample = sample
        pyx_str, times = struct_generator(sample, ngen, wyck_dic)
        pyx_str = get_relaxed_structures(oracle, pyx_str, verbosity=verbose)
        delta_target, pred_target = get_delta_target(oracle, pyx_str, sample[target])
        pyx_str = [s.to(fmt="cif") if s is not None else "" for s in pyx_str]
        if not verbose:
            d, delta_target = min(delta_target, key=lambda x: x[1])
            pyx_str = [pyx_str[d]]
            pred_target = [pred_target[d]]
        rel_structs[i, :] = pyx_str[:]
        rel_targets[i] = pred_target
        pyx_times[i, :] = times[:]
        save_results(cfg, samples, rel_structs, rel_targets, pyx_times, fptr)
        print(f"Saved sample #{start_idx + i}")
    samples = save_results(cfg, samples, rel_structs, rel_targets, pyx_times, fptr)
    # import pandas as pd
    # res = target_parity(cfg, samples)
    # res = res.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    # plot_candidates(cfg.target, res, fptr.split(".")[0] + ".png")


if __name__ == "__main__":
    relax()
