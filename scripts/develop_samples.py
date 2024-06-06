import smact
from pymatgen.core import Composition
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import smact.screening


BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"


def smact_filter(comp):
    comp_dix = comp.get_el_amt_dict()
    els, counts = zip(*[(k, v) for k, v in comp_dix.items()])
    els = [smact.Element(e) for e in els]
    counts = [[int(c)] for c in counts]
    valid = smact.screening.smact_filter(els=els, stoichs=counts)
    return len(valid) > 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_file")
    parser.add_argument("--smact_filter", action="store_true")
    args = parser.parse_args()
    csv_path = DATA_PATH / args.csv_file
    smact_flag = args.smact_filter
    data = pd.read_csv(csv_path, usecols=["readable", "energies"])
    data = data.sort_values(by="energies")
    data[["Stage", "SG", "Composition", "lattice"]] = data["readable"].str.split(
        ";", expand=True
    )
    data["SG"] = data.apply(lambda x: int(x["SG"].split("|")[0]), axis=1)
    data = data.drop(columns=["readable", "Stage"])
    data.to_csv(DATA_PATH / f"sorted_{args.csv_file}")

    if smact_flag:
        data["comp"] = data["Composition"].map(Composition)
        data["SMACT"] = data["comp"].map(smact_filter)
        data = data[data["SMACT"]]
        data = data.drop(columns=["comp", "SMACT"])
        data.to_csv(DATA_PATH / f"filter_{args.csv_file}")
