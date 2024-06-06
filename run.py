from simoraclum.utils.load import sample_candidates, struct_generator
from simoralcum.oracles.mlff import MLOracle
import hydra
from omegaconf import DictConfig
from pathlib import Path

BASE_PATH = Path(__file__).parent
CONFIG_PATH = BASE_PATH / "config"


def get_relaxed_structures(oracle, structs, verbose):
    rel_str = [oracle.relax(s, verbose) for s in structs]
    return rel_str


def get_deviation(oracle, structs, true_vals):
    delta = [(i, abs(oracle.predict(s) - true_vals[i])) for i, s in enumerate(structs)]
    return delta


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="mlff")
def relax(cfg: DictConfig):
    data_file, num, rnd = (cfg.data_file, cfg.nsamples, cfg.random)
    samples = sample_candidates(data_file, num, rnd)
    oracle = MLOracle(cfg.mlff, cfg.target, cfg.rel_iter)
    for s, sample in samples.iterrows():
        pyx_str = struct_generator(sample, cfg.ngen)
        pyx_str = get_relaxed_structures(oracle, pyx_str, verbosity=cfg.verbose)
        delta_target = get_deviation(oracle, pyx_str)


if __name__ == "__main__":
    relax()
