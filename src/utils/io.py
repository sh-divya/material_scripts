import io
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from src.utils.misc import timeout

ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

from contextlib import redirect_stdout, contextmanager

@contextmanager
def optional_redirect(verbose: bool=False):
    if not verbose:
        string_io = io.StringIO()
        with redirect_stdout(string_io):
            yield
    else:
        yield


def sample_candidates(data, num=None, start=0, random=False, gen=False):
    if gen:
        samples = pd.read_csv(DATA_PATH / data, index_col=0)
        if random:
            # data = data.iloc[: num * 10]
            samples = samples.sample(n=num, axis=0)
        else:
            samples = samples.iloc[start:start + num]
        samples = [(start + i, row[1]) for i, row in enumerate(samples.iterrows())]
    else:
        structures = read_structures(data, num, start)
        num = len(structures)
        idx = [start + i for i in range(num)]
        samples = [{"idx": i, "structs": [s], "target": None} for i, s in zip(idx, structures)]
    return samples


def save_results(config, pred_str, pred_targ, times, fptr):
    data = pd.DataFrame()
    fpath = ROOT_PATH / "results"
    fpath.mkdir(parents=True, exist_ok=True)
    start = config["data"]["start"]
    idx = [start + i for i in range(len(pred_str))]
    data["ID"] = idx

    pred_str = pred_str.T
    pred_targ = pred_targ.T
    times = times.T
    for i, (s, t, delta) in enumerate(zip(pred_str, pred_targ, times)):
        targ_col = f"{config['data']['target']}_rel{i}"
        str_col = f"Struct_rel{i}"
        data[str_col] = s
        data[targ_col] = t
        if config["pyxtal"]:
            if config["pyxtal"].get("timer", None):
                time_col = f"time_gen{i}"
                data[time_col] = delta

    # print(fpath / fptr, 3)
    data.to_csv(fpath / fptr)
    return data


def get_saved_results(fptr, target, ngen, num):
    res_path = ROOT_PATH / "results" / fptr
    if res_path.is_file():
        samples = pd.read_csv(res_path, index_col=0)
        all_cols = samples.columns
        struct_cols = []
        targ_cols = []
        time_cols = []
        for col in all_cols:
            try:
                pre, num = str(col).split("_")
                if pre == "Struct":
                    struct_cols.append(str(col))
                elif pre == target:
                    targ_cols.append(str(col))
                elif pre == "time":
                    time_cols.append(str(col))
            except ValueError:
                continue
        rel_structs = samples[struct_cols].values.astype("object")
        rel_targets = samples[targ_cols].values
        gen_times = samples[time_cols].values
    else:
        rel_structs = np.array(
            [[f"{i}{j}_str" for j in range(ngen)] for i in range(num)], dtype="object"
        )
        rel_targets = np.array(
            [[f"{j}{i}_targ" for j in range(ngen)] for i in range(num)]
        )
        gen_times = np.array(
            [[0 for j in range(ngen)] for i in range(num)], dtype="object"
        )
    return rel_structs, rel_targets, gen_times

def read_structures(path, num=None, start=0):
    cif_sources = []
    path = Path(path)
    if path.is_file():
        ext = path.suffix.lower()
        if ext == ".txt":
            with path.open("r") as f:
                cif_paths = [line.strip() for line in f if line.strip()]
            for cif_path in cif_paths:
                cif_path = Path(cif_path)
                assert cif_path.is_file() and cif_path.suffix == ".cif", f"Invalid CIF path: {cif_path}"
                cif_sources.append(cif_path)
        elif ext == ".csv":
            df = pd.read_csv(path)
            assert "cif" in df.columns, "CSV must have a 'cif' column"
            for cif_str in df["cif"]:
                cif_sources.append(str(cif_str))
        elif ext == ".cif":
            cif_sources.append(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    elif path.is_dir():
        for fpath in path.iterdir():
            if fpath.is_file() and fpath.suffix == ".cif":
                cif_sources.append(fpath)

    else:
        raise ValueError(f"Path does not exist: {path}")
    if num is not None:
        cif_sources.sort(key=lambda x: int(x.stem))
        cif_sources = cif_sources[start:start + num]

    structures = []
    for cif in cif_sources:
        try:
            if isinstance(cif, Path):
                struct = Structure.from_file(str(cif))
            else:
                struct = Structure.from_str(str(cif), fmt="cif")
            # SGA = SpacegroupAnalyzer(struc)
            # struc = SGA.get_conventional_standard_structure()
            structures.append(struct.to_conventional())
        except Exception as e:
            print(f"Error reading structure: {e}")
    return structures

def save_results_dix(dix, fptr):
    with open(str(fptr), 'w') as fobj:
        json.dump(dix, fobj)