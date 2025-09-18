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
    try:
        wyck = []
        parse_row = state["Wyckoff"].split("-")
        for item in parse_row:
            item = item.strip("()").split(",")
            z = item[0]
            w = int(item[1])
            wyck.append((z, w))
    except KeyError:
        wyck = None
    lattice = [state[l] for l in ["a", "b", "c", "alpha", "beta", "gamma"]]
    lattice = Lattice.from_para(*lattice[:3], *lattice[3:])
    return comp, sg, lattice, wyck


def parse_wykoff(wyck, wyck_map):
    if wyck and wyck_map:
        sites = {e[0]: [] for e in wyck}
        for elem, wp in wyck:
            sites[elem].append(wyck_map[wp]["name"])
    else:
        sites = None
    return sites


def sample_candidates(data, num, random):
    data = pd.read_csv(DATA_PATH / data, index_col=0)
    if random:
        # data = data.iloc[: num * 10]
        data = data.sample(n=num, axis=0)
    else:
        data = data.iloc[:num]
    return data


@timeout(180)
def sample_pyx(sg, elems, stoich, lattice, sites=None):
    gen = pyxtal()
    gen.from_random(3, sg, elems, stoich, lattice=lattice, sites=None)
    struct = gen.to_pymatgen()
    return struct


def struct_generator(state, ng, wyckoff_map):
    comp, sg, lattice, wyck = parse_state(state)
    wyck = parse_wykoff(wyck, wyckoff_map)
    elems, stoich = zip(*[(k, v) for k, v in comp.items()])
    if wyck:
        wyck = [wyck[e] for e in elems]
    count = 0
    structs = []
    times = []
    print("Starting Gen")
    while count < ng:
        sample = None
        deltaT = 180
        try:
            sample, deltaT = sample_pyx(sg, elems, stoich, lattice, sites=wyck)
            print("Successful sampling")
        except (RuntimeError, TimeoutError) as e:
            count += 1
            structs.append(sample)
            times.append(deltaT)
            if count > ng:
                print("Stopping pyxtal tries because of timeout")
                break
            else:
                print("Issue with Pyxtal, retrying ....")
                continue
        except pyx.msg.Comp_CompatibilityError:
            print("Composition Compatibility")
            structs.extend([sample] * ng)
            times.extend([deltaT] * ng)
            break
        except Exception as e:
            print(e)
            print("Unknown Error")
            pass
        count += 1
        structs.append(sample)
        times.append(deltaT)
    return structs, times


def save_results(config, data, pred_str, pred_targ, times, fptr):
    fpath = ROOT_PATH / "results"
    fpath.mkdir(parents=True, exist_ok=True)
    pred_str = pred_str.T
    pred_targ = pred_targ.T
    times = times.T
    for i, (s, t, delta) in enumerate(zip(pred_str, pred_targ, times)):
        targ_col = f"{config.target}_rel{i}"
        str_col = f"Struct_rel{i}"
        data[str_col] = s
        data[targ_col] = t
        if config.timer:
            time_col = f"time_gen{i}"
            data[time_col] = delta
    data.to_csv(fpath / fptr)
    return data


def get_saved_results(fptr, target, ngen, num):
    res_path = ROOT_PATH / "results" / fptr
    if res_path.is_file():
        samples = pd.read_csv(res_path, index_col=0)
        all_cols = samples.columns[9:]
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
