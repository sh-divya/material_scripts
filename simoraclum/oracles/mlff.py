import matgl


class MLOracle(object):
    def __init__(self, model, target, iters):
        self.model_name = model
        self.target = target
        if self.model == "m3gnet":
            if self.target == "Eform":
                self.model = matgl.load("M3GNet-MP-2018.6.1-Eform")
            elif self.target == "BG":
                self.model = matgl.load("MEGNet-MP-2019.4.1-BandGap-mfi")
        self.niter = iters

    def relax(self, struct, verbose):
        if self.model_name == "m3gnet":
            return struct.relax(verbose=verbose, steps=self.niter)

    def predict(self, struct):
        return self.model.predict_structure(struct)
