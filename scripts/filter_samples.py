import pickle
import smact
from pathlib import Path
from argparse import ArgumentParser


BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"


parser = ArgumentParser()
parser.add_argument("--pkl_file")
args = parser.parse_args()
pkl_path = DATA_PATH / args.pkl_file
data = pickle.load(open(pkl_path, "rb"))

print(data.keys())
