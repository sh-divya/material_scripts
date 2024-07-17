import numpy as np
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
    print("Parsing")
    comp = Composition(state["Composition"]).get_el_amt_dict()
    sg = int(state["SG"])
    lattice = [state[l] for l in ["a", "b", "c", "alpha", "beta", "gamma"]]
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
    print("Starting Gen")
    while count < ng:
        sample = None
        try:
            sample = sample_pyx(sg, elems, stoich, lattice)
            print("Successful sampling")
        except (RuntimeError, TimeoutError) as e:
            count += 1
            structs.append(sample)
            if count > ng:
                print("Stopping pyxtal tries because of timeout")
                break
            else:
                print("Issue with Pyxtal, retrying ....")
                continue
        except pyx.msg.Comp_CompatibilityError:
            print("Composition Compatibility")
            structs.extend([sample] * ng)
            break
        except Exception as e:
            print(e)
            print("Unknown Error")
            pass
        count += 1
        structs.append(sample)
    return structs


def save_results(config, data, pred_str, pred_targ, fptr):
    fpath = ROOT_PATH / "results"
    fpath.mkdir(parents=True, exist_ok=True)
    pred_str = pred_str.T
    pred_targ = pred_targ.T
    for i, (s, t) in enumerate(zip(pred_str, pred_targ)):
        targ_col = f"{config.target}_rel{i}"
        str_col = f"Struct_rel{i}"
        data[str_col] = s
        data[targ_col] = t
    data.to_csv(fpath / fptr)
    return data


def get_saved_results(fptr, ngen, num):
    res_path = ROOT_PATH / "results" / fptr
    if res_path.is_file():
        samples = pd.read_csv(res_path, index_col=0)
        all_cols = samples.columns[9:]
        struct_cols = all_cols[0::2]
        targ_cols = all_cols[1::2]
        rel_structs = samples[struct_cols].values.astype(str)
        rel_targets = samples[targ_cols].values
    else:
        rel_structs = np.array(
            [[f"{i}{j}_str" for j in range(ngen)] for i in range(num)]
        )
        rel_targets = np.array(
            [[f"{j}{i}_targ" for j in range(ngen)] for i in range(num)]
        )
    return rel_structs, rel_targets
