import io
import contextlib
from pathlib import Path
from yaml import safe_load
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Union

import torch
import matgl
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE
from mace.calculators import mace_mp
from matgl.ext.ase import Relaxer
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.utils.io import optional_redirect

MATGL_VERSION = "1.3.0"
MACE_VERSION = "0.3.14"

MP_REFERENCES = safe_load(open(str(Path(__file__).parent.parent.parent / "config" / "elements.yaml")))

class MLOracle(object):
    def __init__(self, model, target, iters):
        self.model_name = model
        self.target = target

        opt_args = {"max_steps": iters, "cell_only": "Frechet"}
        if self.model_name == "m3gnet":
            matgl.clear_cache(confirm=False)
            if self.target == "Eform":
                # self.model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
                self.model = AtomicGraph("m3gnet", "M3GNet-MP-2018.6.1-Eform", relax=opt_args, target=target)
            elif self.target == "BG":
                self.model = AtomicGraph("m3gnet", "MEGNet-MP-2019.4.1-BandGap-mfi", relax=opt_args, target=target)
                # self.model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
        elif self.model_name == "mace":
                self.model = AtomicGraph("mace", "small", relax=opt_args, target=target)

    def relax(self, struct, verbose):
        # if self.model_name == "m3gnet":
        #     # return struct.relax(verbose=verbose, steps=self.niter)
        #     return struct.relax(verbose=True, steps=self.niter)
        return self.model.minimize(struct, verbose)

    def predict(self, struct):
        # return self.model.predict_structure(struct, False)
        return self.model(struct, False)


class MACEPredict:
    """
    Wrapper around MACE ASE calculator to predict total energy.

    Parameters
    ----------
    model : str
        The model name or checkpoint identifier for the pretrained MACE model.
    relax_cell : bool
        Whether cell relaxation is allowed.
    device : str
        Device type ('cpu' or 'cuda') for model execution.
    dtype : str, optional
        Floating point precision type (default is "float64").
    """

    def __init__(self, model: str, relax_cell: bool, device: str, dtype="float64"):
        self.model = mace_mp(model, device, dtype)
        self.relax_cell = relax_cell

    def predict_structure(self, sample: Structure) -> float:
        """
        Predicts formation energy per atom.

        Parameters
        ----------
        sample : Structure
            Pymatgen structure for which to compute the formation energy.

        Returns
        -------
        float
            Formation energy per atom.
        """
        te = sample.calculate(self.model).get_potential_energy()
        elements = sample.composition.get_el_amt_dict()
        eref = sum([c * MP_REFERENCES[e] for e, c in elements.items()])
        return (te - eref) / len(list(sample))

    def relax(
        self,
        sample: Structure,
        fmax: float = 0.1,
        ase_cellfilter: str = "Exp",
        steps: int = 10,
        verbose: bool = False
    ) -> Structure:
        """
        Performs structural relaxation.

        Parameters
        ----------
        sample : Structure
            Structure to relax.
        fmax : float, optional
            Force convergence criterion.
        ase_cellfilter : str, optional
            Type of cell constraint to apply ("Exp" or "Frechet").
        steps : int, optional
            Maximum number of relaxation steps.

        Returns
        -------
        Structure
            Relaxed structure.
        """
        # redirect unnecesary printing


        # with redirect_stdout(io.StringIO()):
        with optional_redirect(verbose):
            if self.relax_cell:
                # serial geometry relaxation through ASE
                sample = AseAtomsAdaptor.get_atoms(sample)
                sample.calc = self.model
                if ase_cellfilter == "Exp":
                    sample = ExpCellFilter(sample)
                elif ase_cellfilter == "Frechet":
                    sample = FrechetCellFilter(sample)
                else:
                    raise ValueError(
                        f"{ase_cellfilter} not available as constraint for relaxation"
                    )
                opt = FIRE(sample, logfile="-")
                opt.run(fmax=fmax, steps=steps)
                sample = AseAtomsAdaptor.get_structure(sample.atoms)
                return sample
            else:
                return sample.relax(self.model, fmax=fmax, steps=steps)


class AtomicGraph(object):
    """
    GFlowNet proxy that uses crystal graph models (M3GNet or MACE) to predict material properties.

    Parameters
    ----------
    model : str
        Name of the model to use, either 'm3gnet' or 'mace'.
    ckpt_path : str, optional
        Path or identifier of the pretrained model checkpoint.
    relax : bool or str, optional
        Whether to apply structure relaxation. If str, specifies a relaxation model checkpoint.
    target : str, optional
        Property to predict, e.g., 'eform' (formation energy).
    **kwargs :
        Additional arguments for the Proxy base class.
    """

    def __init__(
        self,
        model: str,
        ckpt_path: Optional[str] = None,
        relax: Union[bool, dict] = False,
        target: str = "eform",
    ):

        self.model_name = model

        self.relax = relax
        # Will inform the class variable self.constraint
        # but needs to be boolean for initializing the relaxer
        cell_only = "Exp"

        if self.relax:
            if isinstance(self.relax, bool):
                # if relax parameters are not specified, defaults values are assumed
                # except for relax model, which depends on the property model
                self.relax = {"fmax": 0.01, "max_steps": 100, "cell_only": True}
            self.opt_args = {
                "fmax": self.relax.get("fmax", 0.01),
                "max_steps": self.relax.get("max_steps", 100),
            }

            # cell_only can be specified with specific ASE contraint type, or bool
            cell_only = self.relax.get("cell_only", "Exp")
            if isinstance(cell_only, str):
                # save constraint type, and change cell_only to boolean
                self.constraints = cell_only
                cell_only = True
            else:
                # in case cell_only is boolean and True, give default value to constraint
                self.constraints = "Exp"

        if model == "m3gnet":
            try:
                matgl_version = version("matgl")
            except PackageNotFoundError:
                print(" `matgl` cannot be imported.")
                raise PackageNotFoundError("Matgl not found")

            if matgl_version != MATGL_VERSION:
                print(f"Currentl matgl version {matgl_version}")
                print(f"Require matgl version {MATGL_VERSION}")
                raise ImportError("Version Mismatch")
            if not ckpt_path:
                if target == "eform":
                    ckpt_path = "M3GNet-MP-2018.6.1-Eform"
                # elif target == "bg":
                #     ckpt_path = "MEGNet-MP-2019.4.1-BandGap-mfi"
                else:
                    raise ValueError(
                        f"{target} not available as pretrained model in m3gnet"
                    )
            self.model = matgl.load_model(ckpt_path)
            if self.relax:
                self.relax = self.relax.get("model", "M3GNet-MP-2021.2.8-PES")
                self.relax = matgl.load_model(self.relax)
                self.relax = Relaxer(self.relax, relax_cell=cell_only)
        elif model == "mace":
            try:
                mace_version = version("mace-torch")
            except PackageNotFoundError:
                print(" `mace` cannot be imported.")
                raise PackageNotFoundError("mace not found")

            if mace_version != MACE_VERSION:
                print(f"Currentl mace version {mace_version}")
                print(f"Require mace version {MACE_VERSION}")
                raise ImportError("Version Mismatch")
            if not ckpt_path:
                if target == "eform":
                    ckpt_path = "small"
                else:
                    raise ValueError(
                        f"{target} not available as pretrained model in m3gnet"
                    )
                
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = MACEPredict(
                ckpt_path,
                relax_cell=cell_only,
                device=device,
                dtype="float32"
            )
            if self.relax:
                # Cannot specify differnt models for property prediction
                # and relaxation for MACE
                if isinstance(self.relax, str):
                    raise ValueError(
                        f"{model} uses the same model for relaxation to calculate {target}"
                    )
                else:
                    self.relax = self.model

    def minimize(self, sample, verbose):
        sample = self.relax.relax(
                sample,
                fmax=self.opt_args["fmax"],
                steps=self.opt_args["max_steps"],
                ase_cellfilter=self.constraints,
                verbose=verbose
            )
        if self.model_name == "m3gnet":
            sample = sample["final_structure"]
        
        return sample


    def __call__(self, sample: Structure, relax: bool) -> float:
        """
        Predict the property (e.g., formation energy) for a given structure.

        Parameters
        ----------
        sample : pymatgen.Structure
            Atomic structure to evaluate.

        Returns
        -------
        float
            Predicted property value.
        """
        if relax:
            sample = self.minimize(sample)
        target = self.model.predict_structure(sample)
        try:
            target = target.item()
        except AttributeError:
            pass
        return target
