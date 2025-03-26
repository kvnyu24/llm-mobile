#!/bin/bash

# Run Layer Compression and Skipping Example
#
# This script provides a convenient way to run the layer compression example
# with different parameter configurations.

# Set path to Python (use virtual environment if available)
if [ -d "venv" ]; then
    PYTHON_PATH="venv/bin/python"
else
    PYTHON_PATH="python"
fi

# Default values
MODEL="gpt2"
RANK=8
THRESHOLD=0.3
COMPUTE=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --compute)
            COMPUTE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL       HuggingFace model name (default: gpt2)"
            echo "  --rank RANK         Compression rank (default: 8)"
            echo "  --threshold THRESHOLD  Skip threshold (default: 0.3)"
            echo "  --compute COMPUTE   Available compute factor 0-1 (default: 0.5)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running layer compression example with:"
echo "  Model: $MODEL"
echo "  Compression rank: $RANK"
echo "  Skip threshold: $THRESHOLD"
echo "  Available compute: $COMPUTE"
echo ""

# Run the example
$PYTHON_PATH src/examples/layer_compression_example.py \
    --model-name "$MODEL" \
    --compression-rank "$RANK" \
    --skip-threshold "$THRESHOLD" \
    --available-compute "$COMPUTE" 