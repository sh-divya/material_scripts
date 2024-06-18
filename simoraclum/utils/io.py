import pandas as pd
from pathlib import Path
import pyxtal as pyx
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pymatgen.core import Composition
from simoraclum.utils.misc import timeout

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


@timeout(180)
def sample_pyx(sg, elems, stoich, lattice):
    gen = pyxtal()
    gen.from_random(3, sg, elems, stoich, lattice=lattice)
    struct = gen.to_pymatgen()
    return struct


# import random
# import time
# @timeout(5)
# def sample_pyx(sg, elems, stoich, lattice):
#     n = random.randint(1, 10)
#     time.sleep(n)
#     return None


def struct_generator(state, ng):
    comp, sg, lattice = parse_state(state)
    elems, stoich = zip(*[(k, v) for k, v in comp.items()])
    count = 0
    structs = []
    while count < ng:
        sample = None
        tries = 0
        try:
            sample = sample_pyx(sg, elems, stoich, lattice)
            print("Successful sampling")
        except (RuntimeError, TimeoutError) as e:
            tries += 1
            if tries > 20:
                print("Stopping pyxtal tries because of timeout")
                pass
            else:
                print("Issue with Pyxtal, retrying ....")
            continue
        except pyx.msg.Comp_CompatibilityError:
            print("Composition Compatibility")
            pass
        count += 1
        structs.append(sample)
    return structs


def save_results(config, data, pred_str, pred_targ):
    fptr = config.data_file.split(".")[0]
    v = "_verbose" if config.verbose else ""
    fptr = f"{fptr}{config.nsamples}_ngen{config.ngen}_steps{config.rel_iter}_{config.mlff}{v}.csv"
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
    return data, str(fpath / fptr)
