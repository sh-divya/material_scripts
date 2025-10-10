from abc import ABC
import numpy as np

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty


CnnFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

class BaseMetric(ABC):
    def __init__(self, name):
        self.name = name
        self.results = None

    def write(self):
        pass

    def single(self, s, r):
        pass

    def compute(self, samples, reference):
        pass


class COV(BaseMetric):
    def __init__(self, name):
        super().__init__(self, name)
        self.struc_thresh = 0.4
        self.comp_thresh = 10

    def write(self):
        print(self.results)
    
    def compute(self, distances):
        struct_pdist = distances["struct"]
        comp_pdist = distances["comp"]
        rec = np.mean(np.logical_and(struct_pdist.min(axis=0), comp_pdist.min(axis=0)))

def fingerprints(crystals):
    values = {"idx": [], "cnn": [], "magpie": [], "invalid": []}
    for s, struct in enumerate(crystals):
        comp = struct.composition
        comp_fp = CompFP.featurize(comp)
        # print(comp_fp)
        try:
            site_fps = [CnnFP.featurize(struct, i) for i in range(len(struct))]
            # print(site_fps)
        except Exception as e:
            print(e)
            values["invalid"].append(s)
        values["idx"].append(s)
        values["cnn"].append(site_fps)
        values["magpie"].append(comp_fp)

    return values
