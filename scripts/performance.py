import pandas as pd
from pathlib import Path
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