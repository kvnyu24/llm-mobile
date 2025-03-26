#!/bin/bash

# Script to run token pruning with different configurations

# Set the path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default prompt
PROMPT="Once upon a time in a land far away, there was a kingdom ruled by a wise and just queen. The people of the kingdom were happy and prosperous under her rule."

# Run with different thresholds
echo "=== Running with different pruning thresholds ==="
python src/examples/token_pruning_example.py --prompt "$PROMPT" --max_tokens 50 --threshold 0.005
echo ""
python src/examples/token_pruning_example.py --prompt "$PROMPT" --max_tokens 50 --threshold 0.01
echo ""
python src/examples/token_pruning_example.py --prompt "$PROMPT" --max_tokens 50 --threshold 0.02
echo ""

# Run with longer generation
echo "=== Running with longer generation ==="
python src/examples/token_pruning_example.py --prompt "$PROMPT" --max_tokens 100 --threshold 0.01
echo ""

# Run without pruning for comparison
echo "=== Running without pruning ==="
python src/examples/token_pruning_example.py --prompt "$PROMPT" --max_tokens 50 --no_pruning
echo ""

# Run with GPU if available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "=== Running on GPU ==="
    python src/examples/token_pruning_example.py --prompt "$PROMPT" --max_tokens 50 --threshold 0.01 --device cuda
fi 