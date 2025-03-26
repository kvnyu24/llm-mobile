#!/bin/bash

# Run Memory Manager Example
#
# This script provides a convenient way to run the memory manager example
# with different parameter configurations.

# Set path to Python (use virtual environment if available)
if [ -d "venv" ]; then
    PYTHON_PATH="venv/bin/python"
else
    PYTHON_PATH="python"
fi

# Default values
MAX_MEMORY=50
PAGE_SIZE=16
THRESHOLD=90
QUANTIZATION="--quantization"
OFFLOADING="--offloading"
FORCE_EVICTION="--force-eviction"
USE_MODEL=""
MODEL_NAME="gpt2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --memory)
            MAX_MEMORY="$2"
            shift 2
            ;;
        --page-size)
            PAGE_SIZE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --no-quantization)
            QUANTIZATION=""
            shift
            ;;
        --no-offloading)
            OFFLOADING=""
            shift
            ;;
        --no-eviction)
            FORCE_EVICTION=""
            shift
            ;;
        --use-model)
            USE_MODEL="--use-model"
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --memory MB        Maximum memory budget in MB (default: 50)"
            echo "  --page-size SIZE   Number of tokens per page (default: 16)"
            echo "  --threshold PCT    Memory threshold percentage (default: 90)"
            echo "  --no-quantization  Disable quantization compression"
            echo "  --no-offloading    Disable offloading to cloud/disk"
            echo "  --no-eviction      Disable forced eviction"
            echo "  --use-model        Use actual HuggingFace model"
            echo "  --model NAME       HuggingFace model name (default: gpt2)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running memory manager example with:"
echo "  Maximum memory: $MAX_MEMORY MB"
echo "  Page size: $PAGE_SIZE tokens"
echo "  Memory threshold: $THRESHOLD%"
echo "  Quantization: ${QUANTIZATION:+enabled}"
echo "  Offloading: ${OFFLOADING:+enabled}"
echo "  Force eviction: ${FORCE_EVICTION:+enabled}"
if [ -n "$USE_MODEL" ]; then
    echo "  Using model: $MODEL_NAME"
fi
echo ""

# Run the example
$PYTHON_PATH src/examples/memory_manager_example.py \
    --max-memory "$MAX_MEMORY" \
    --page-size "$PAGE_SIZE" \
    --threshold "$THRESHOLD" \
    $QUANTIZATION \
    $OFFLOADING \
    $FORCE_EVICTION \
    $USE_MODEL \
    --model-name "$MODEL_NAME" 