#!/bin/bash
set -x

python generate_nshot_cot_exers.py && python eval_simple.py
python generate_nshot_direct_exers.py && python eval_simple.py
python generate_0shot_cot_exers.py && python eval_simple.py
python generate_0shot_direct_exers.py && python eval_simple.py

python generate_nshot_cot_exers.py && python eval_simple.py with_labels
python generate_nshot_direct_exers.py && python eval_simple.py with_labels
python generate_0shot_cot_exers.py && python eval_simple.py with_labels
python generate_0shot_direct_exers.py && python eval_simple.py with_labels

python generate_nshot_cot_exers.py && python eval.py
python generate_nshot_direct_exers.py && python eval.py
python generate_0shot_cot_exers.py && python eval.py
python generate_0shot_direct_exers.py && python eval.py
