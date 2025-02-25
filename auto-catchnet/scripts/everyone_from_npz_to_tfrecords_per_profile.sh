#!/bin/bash

export PYTHONPATH="$(pwd)"

npz_path="/home/chy/archive-data/processed/everyone-npz"
out_path="/home/chy/archive-data/processed/everyone-tfr-profile"

# ulimit -n 4096  (profile 개수가 1400개 정도 됨)

python -m ai.dataset.everyone.tfrecord.from_npz_per_profile --npz_path=${npz_path} --out_path=${out_path}
