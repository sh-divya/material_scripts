## Introduction

Repository to use either ML or QM based crystal relaxation oracles. Currently implements generating random initializations of a crystal state (composition, space group and lattice parameters) and relaxing them using M3gnet so as to choose structures that closely match the target (either formation energy or bandgap) as predicted by the [DAVE](https://github.com/sh-divya/ActiveLearningMaterials) proxy.

## Example Setup Solution

1. Conda environment with `python==3.10`
2. `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
3. `conda install lightning -c conda-forge`
4. `conda install pip`
5. `python -m pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu118.html`
6. `python -m pip install pydantic`
7. `pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html`
8. `python -m pip install --no-dependencies matgl`
9. `conda install pandas`
10. `conda install click`
11. `conda install --channel conda-forge pymatgen`
12. `python -m importlib-metadata==1.4`
13. `python -m pip install --no-dependencies pyxtal`
14. `python -m pip install smact`
15. `python -m pip install mendeleev`

## Usage

For an example training data file "train_data.csv" which should be present in a directory called `data`, the following command should be run
`python run.py --config-name=train_data`
It will generate relaxed pyxtal samples and predicted energies in a direcotry called `results`, which should be created before running the command


