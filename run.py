from simoraclum.utils.io import sample_candidates, struct_generator, save_results
from simoraclum.utils.post import target_parity
from simoraclum.oracles.mlff import MLOracle
import hydra
from omegaconf import DictConfig
from pathlib import Path

BASE_PATH = Path(__file__).parent
CONFIG_PATH = BASE_PATH / "config"


def get_relaxed_structures(oracle, structs, verbosity):
    rel_str = [oracle.relax(s, verbosity).to(fmt="cif") for s in structs]
    return rel_str


def get_delta_target(oracle, structs, true_val):
    pred_vals = [oracle.predict(s) for s in structs]
    delta = [(i, abs(val - true_val)) for i, val in pred_vals]
    return delta, pred_vals


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="mlff")
def relax(cfg: DictConfig):
    data_file, num, rnd = (cfg.data_file, cfg.nsamples, cfg.random)
    samples = sample_candidates(data_file, num, rnd)
    oracle = MLOracle(cfg.mlff, cfg.target, cfg.rel_iter)
    rel_structs = []
    rel_targets = []
    for s, sample in samples.iterrows():
        pyx_str = struct_generator(sample, cfg.ngen)
        pyx_str = get_relaxed_structures(oracle, pyx_str, verbosity=cfg.verbose)
        delta_target, pred_target = get_delta_target(
            oracle, pyx_str, sample["energies"]
        )
        if not cfg.verbose:
            d, delta_target = min(delta_target, key=lambda x: x[1])
            pyx_str = [pyx_str[d]]
            pred_target = [pred_target[d]]
        rel_structs.append(pyx_str)
        rel_targets.append(pred_target)
    save_results(cfg, samples, rel_structs, rel_targets)


if __name__ == "__main__":
    relax()
