import argparse

import torch

from .data import append_movie_list, get_movie_list, split_data, write_data
from .train import train

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

parser = argparse.ArgumentParser(prog="movie-names")
parser.add_argument("apikey")

args = parser.parse_args()

# have, want = get_movie_list(args.apikey)
#append_movie_list(args.apikey, "data/movie_names")
# write_data(have, want, "data/movie_names")
#split_data("data/movie_names")
train()
