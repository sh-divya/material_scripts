import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd

from src.utils.io import read_structures, save_results_dix
from src.benchmarks.base import fingerprints

warnings.simplefilter("ignore", UserWarning)
BASE_PATH = Path(__file__).parent
CONFIG_PATH = BASE_PATH / "config" / "benchmark"
RESULTS_PATH = BASE_PATH / "results"


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="mp20")
def run(cfg: DictConfig):
    ref, data_path, task, num = cfg.ref, cfg.data_path, cfg.task, cfg.num
    if not num:
        num_str = 'full'
    else:
        num_str = num
    samples_name = f"{task}_samples_{num_str}"
    ref_name = f"{task}_{ref}"
    samples_file = RESULTS_PATH / f"{samples_name}.json"
    ref_file = RESULTS_PATH / f"{ref_name}.json"

    # Read samples structures if needed
    if samples_file.exists():
        print(f"Samples file {samples_file} exists. Skipping reading samples.")
        samples_structures = None
    else:
        print(f"Samples file {samples_file} not found. Reading samples from {data_path['samples']}...")
        samples_structures = read_structures(data_path["samples"], num=num)
        print(f"Read {len(samples_structures)} samples structures.")
        sample_fp = fingerprints(samples_structures)
        save_results_dix(sample_fp, samples_file)

    # Read test structures if needed
    if ref_file.exists():
        print(f"Reference file {ref_file} exists. Skipping reading test structures.")
        test_structures = None
    else:
        print(f"Reference file {ref_file} not found. Reading test structures from {data_path['test']}...")
        test_structures = read_structures(data_path["test"])
        print(f"Read {len(test_structures)} test structures.")
        ref_fp = fingerprints(test_structures)
        save_results_dix(ref_fp, ref_file)

if __name__ == "__main__":
    run()
