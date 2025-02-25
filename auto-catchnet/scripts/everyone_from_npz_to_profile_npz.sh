#!/bin/bash

export PYTHONPATH="$(pwd)"

npz_path="/home/chy/archive-data/processed/everyone-npz"
out_path="/home/chy/archive-data/processed/everyone-npz-profile"

python -m ds.everyone.npz.from_npz_split_profile --npz_path=${npz_path} --out_path=${out_path}
