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
    elems, count = zip(*[(k, v) for k, v in comp.items()])
    count = 0
    structs = []
    while count < ng:
        try:
            gen.from_random(3, sg, elems, count, lattice)
        except RuntimeError:
            continue
        except pyx.msg.Comp_CompatibilityError:
            break
        count += 1
        structs.append(gen.to_pymatgen())
    return structs
