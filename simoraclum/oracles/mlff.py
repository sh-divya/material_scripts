import matgl


class MLOracle(object):
    def __init__(self, model, target, iters):
        self.model_name = model
        self.target = target
        if self.model_name == "m3gnet":
            matgl.clear_cache(confirm=False)
            if self.target == "Eform":
                self.model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
            elif self.target == "BG":
                self.model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
        self.niter = iters

    def relax(self, struct, verbose):
        if self.model_name == "m3gnet":
            return struct.relax(verbose=verbose, steps=self.niter)

    def predict(self, struct):
        return self.model.predict_structure(struct)
