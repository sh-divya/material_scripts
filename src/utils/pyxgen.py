import errno
import os
import signal
import functools
import time

import pyxtal as pyx
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pymatgen.core import Composition


# adapted from https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            delta = seconds
            try:
                t1 = time.time()
                results = func(*args, **kwargs)
                t2 = time.time()
                delta = t2 - t1
            except KeyError:
                pass
            finally:
                signal.alarm(0)

            return results, delta

        return wrapper

    return decorator


def parse_state(state):
    print("Parsing")
    try:
        comp = Composition(state["Composition"])
    except KeyError:
        comp = Composition(state["Formulae"])

    comp = comp.get_el_amt_dict()
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
