import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from src.utils.io import (
    sample_candidates,
    save_results,
    get_saved_results,
)

from src.utils.pyxgen import struct_generator
# from src.utils.post import target_parity, plot_candidates
from src.oracles.mlff import MLOracle
from src.oracles.common import get_delta_target, get_relaxed_structures

# from pymatgen.core import Structure
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from yaml import safe_load
import numpy as np

BASE_PATH = Path(__file__).parent
CONFIG_PATH = BASE_PATH / "config" / "relax"


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="mlff")
def relax(cfg: DictConfig):
    # Convert config and extract main components
    cfg = OmegaConf.to_container(cfg)
    print("Job Config:")
    print("-----------")
    print(cfg)
    print("-----------")

    data, relcfg, gen = cfg["data"], cfg["relax"], cfg.get("pyxtal", None)
    try:
        ngen = gen.get("ngen", 1)
    except AttributeError:
        ngen = 1

    # Initialize configuration variables
    wyckoff = False
    verbose = relcfg.get("verbose", False)
    target = data["target"]
    start_idx = int(data.get("start", 0))
    data["start"] = start_idx

    # Set up file path and naming
    fptr = cfg["name"]
    # print(fptr, 1)
    v = "_verbose" if verbose else ""
    w = "_wyck" if wyckoff else ""

    # Sample and oracle initialization
    samples = sample_candidates(
        BASE_PATH / "data" / data["src"], data.get("nsamples", None), start_idx, gen
    )

    num = len(samples)
    start_str = f"start{start_idx}-" if start_idx else ""
    fptr = (
        f"{fptr}_{start_str}{num}_ngen{ngen}_steps{relcfg['rel_iter']}_{relcfg['mlff']}{w}{v}.csv"
    )
    # print(fptr, 2)
    oracle = MLOracle(relcfg["mlff"], target, relcfg["rel_iter"])

    # Initialize arrays for results
    # rel_structs = np.empty((num, ngen), dtype=object)
    # rel_targets = np.empty((num, ngen) if verbose else (num,), dtype=float)
    # pyx_times = np.zeros((num, ngen))
    # times = np.zeros(ngen)

    # Set up Wyckoff configuration if needed
    if gen:
        wyckoff = gen.get("wyckoff", False)
        timer = gen.get("timer", False)
        if wyckoff:
            wyck_dic = safe_load(open(str(CONFIG_PATH / "wyckoff.yaml")))

    rel_structs, rel_targets, pyx_times = get_saved_results(fptr, target, ngen, num)
    # Get saved results and generate structures if needed
    if gen:
        samples, times = struct_generator(samples, ngen, wyck_dic)
    else:
        times = [None for _ in range(ngen)]

    # Process each sample
    for i, sample in tqdm(enumerate(samples)):
        relstr = get_relaxed_structures(oracle, sample['structs'], verbosity=verbose)
        delta_target, pred_target = get_delta_target(oracle, relstr, sample["target"])
        pyx_str = [s.to(fmt="cif") if s is not None else "" for s in relstr]
        
        if not verbose:
            d, delta_target = min(delta_target, key=lambda x: x[1])
            pyx_str = [pyx_str[d]]
            pred_target = [pred_target[d]]
        
        rel_structs[i, :] = pyx_str[:]
        rel_targets[i] = pred_target
        pyx_times[i, :] = times[:]
        save_results(cfg, rel_structs, rel_targets, pyx_times, fptr)
        print(f"Saved sample #{start_idx + i}")
    samples = save_results(cfg, rel_structs, rel_targets, pyx_times, fptr)


if __name__ == "__main__":
    relax()
