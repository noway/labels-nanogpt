#!/bin/bash
set -x

python train_simple.py
python train_simple.py with_labels
python train.py
