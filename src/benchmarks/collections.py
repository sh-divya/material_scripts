from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from src.benchmarks.base import BaseMetric


class CDVAE(BaseMetric):
    def __init__(self, samples, references):
        self.name = "CDVAE"
        self.metrics = {"COV-R": None, "COV-P": None, "AMSD-R": None, "AMSD-P": None, "AMCD-R": None, "AMCD-P": None}

    def distances(self):
        pass

