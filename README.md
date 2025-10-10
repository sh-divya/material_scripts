## Introduction

Repository to use either ML or QM based crystal relaxation oracles. Currently implements generating random initializations of a crystal state (composition, space group and lattice parameters) and relaxing them using M3gnet so as to choose structures that closely match the target (either formation energy or bandgap) as predicted by the [DAVE](https://github.com/sh-divya/ActiveLearningMaterials) proxy.

## Example Setup Solution

1. Python venv with `python==3.10`
2. `pip install torch==2.3.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`
3. `pip install lightning`
5. `pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu118.html`
6. `pip install pydantic`
7. `pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html`
8. `pip install --no-dependencies matgl`
10. `pip install click`
11. `pip install hydra-core --upgrade`
12. `pip install pymatgen`
13. `pip install ase` 
14. `pip install boto3`
15. `pip install torchdata==0.8.0`
16. `python -m importlib-metadata==1.4`
17. `python -m pip install --no-dependencies pyxtal`
18. `python -m pip install smact`
19. `python -m pip install mendeleev`

## Usage

For an example training data file "train_data.csv" which should be present in a directory called `data`, the following command should be run
`python run.py --config-name=train_data`
It will generate relaxed pyxtal samples and predicted energies in a direcotry called `results`, which should be created before running the command


