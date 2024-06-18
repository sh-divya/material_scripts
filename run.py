from simoraclum.utils.io import sample_candidates, struct_generator, save_results
from simoraclum.utils.post import target_parity, plot_candidates
from simoraclum.oracles.mlff import MLOracle

# from pymatgen.core import Structure
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

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
    data_file, num, rnd = (cfg.data_file, cfg.nsamples, cfg.random)
    samples = sample_candidates(data_file, num, rnd)
    oracle = MLOracle(cfg.mlff, cfg.target, cfg.rel_iter)
    rel_structs = [[f"{i}{j}_str" for j in range(cfg.ngen)] for i in range(num)]
    rel_targets = [[f"{j}{i}_targ" for j in range(cfg.ngen)] for i in range(num)]
    for i, sample in tqdm(enumerate(samples.iterrows())):
        s, sample = sample
        pyx_str = struct_generator(sample, cfg.ngen)
        pyx_str = get_relaxed_structures(oracle, pyx_str, verbosity=cfg.verbose)
        delta_target, pred_target = get_delta_target(
            oracle, pyx_str, sample["energies"]
        )
        pyx_str = [s.to(fmt="cif") if s is not None else "" for s in pyx_str]
        if not cfg.verbose:
            d, delta_target = min(delta_target, key=lambda x: x[1])
            pyx_str = [pyx_str[d]]
            pred_target = [pred_target[d]]
        rel_structs[i] = pyx_str
        rel_targets[i] = pred_target
        samples, fptr = save_results(cfg, samples, rel_structs, rel_targets)
        print(f"Saved sample #{i}")
    # import pandas as pd

    # fptr = str(Path("./results/filter_gfn_samples5_ngen3_steps100_m3gnet_verbose.csv"))
    # samples = pd.read_csv(fptr, index_col=0)
    # print(samples)
    # res = target_parity(cfg, samples)
    # plot_candidates(cfg.target, res, fptr.split(".")[0] + ".png")


if __name__ == "__main__":
    relax()
