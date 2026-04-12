#!/bin/bash
set -e

echo "Starting rigorous Grue Training Matrix"

for seed in 1 2 3 4 5; do
    echo "Training Scheme A, Seed $seed"
    python3 train.py --scheme A --seed $seed
    
    echo "Training Scheme B, Seed $seed"
    python3 train.py --scheme B --seed $seed
done

echo "Matrix complete. Evaluating last models..."
python3 evaluate.py --scheme A --seed 5
python3 evaluate.py --scheme B --seed 5

echo "Running Subtraction Analysis..."
python3 subtract_models.py
