#!/bin/bash

export PYTHONPATH="$(pwd)"

python -m ai.nas.utils.cell_vis \
    --conv_arc="0 1 1 0 0 1 0 0" \
    --reduce_arc="0 1 1 0 1 0 0 0"
