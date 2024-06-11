import pandas as pd
from pathlib import Path
import pyxtal as pyx
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pymatgen.core import Composition

ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"


def parse_state(state):
    comp = Composition(state["Composition"]).get_el_amt_dict()
    sg = int(state["SG"])
    lattice = [float(i.strip(" ").strip("()")) for i in state["lattice"].split(",")]
    lattice = Lattice.from_para(*lattice[:3], *lattice[3:])
    return comp, sg, lattice


def sample_candidates(data, num, random):
    data = pd.read_csv(DATA_PATH / data, index_col=0)
    if random:
        data = data.iloc[: num * 10]
        data = data.sample(n=num, axis=0)
    else:
        data = data.iloc[:num]
    return data


def struct_generator(state, ng):
    gen = pyxtal()
    comp, sg, lattice = parse_state(state)
    elems, stoich = zip(*[(k, v) for k, v in comp.items()])
    count = 0
    structs = []
    while count < ng:
        try:
            gen.from_random(3, sg, elems, stoich, lattice=lattice)
        except RuntimeError:
            continue
        except pyx.msg.Comp_CompatibilityError:
            break
        count += 1
        structs.append(gen.to_pymatgen())
    return structs


def save_results(config, data, pred_str, pred_targ):
    fptr = config.data_file.split(".")[0]
    v = "_verbose" if config.verbose else ""
    fptr = f"{fptr}_ngen{config.ngen}_steps{config.rel_iter}_{config.mlff}{v}.csv"
    fpath = ROOT_PATH / "results"
    fpath.mkdir(parents=True, exist_ok=True)
    pred_str = zip(*(pred_str))
    pred_targ = zip(*(pred_targ))
    for i, (s, t) in enumerate(zip(pred_str, pred_targ)):
        targ_col = f"{config.target}_rel{i}"
        str_col = f"Struct_rel{i}"
        data[str_col] = s
        data[targ_col] = t
    data.to_csv(fpath / fptr)
    return data
