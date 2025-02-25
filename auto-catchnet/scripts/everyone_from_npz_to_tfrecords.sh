#!/bin/bash

export PYTHONPATH="$(pwd)"

npz_path="/home/chy/archive-data/processed/everyone-npz"
out_path="/home/chy/archive-data/processed/everyone-tfr-candide"

python -m ai.dataset.everyone.tfrecord.from_npz --npz_path=${npz_path} --out_path=${out_path}
