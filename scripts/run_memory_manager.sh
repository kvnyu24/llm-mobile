#!/bin/bash

# Default parameters
MAX_MEMORY=50
PAGE_SIZE=16
THRESHOLD=90
QUANTIZATION=true
OFFLOADING=true
FORCE_EVICTION=true
LOG_LEVEL="INFO"

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
      QUANTIZATION=false
      shift
      ;;
    --no-offloading)
      OFFLOADING=false
      shift
      ;;
    --no-eviction)
      FORCE_EVICTION=false
      shift
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run_memory_manager.sh [options]"
      echo "Options:"
      echo "  --memory VALUE       Set maximum memory budget in MB (default: 50)"
      echo "  --page-size VALUE    Set number of tokens per page (default: 16)"
      echo "  --threshold VALUE    Set memory threshold percentage (default: 90)"
      echo "  --no-quantization    Disable quantization compression"
      echo "  --no-offloading      Disable offloading to cloud/disk"
      echo "  --no-eviction        Disable forced eviction"
      echo "  --log-level LEVEL    Set logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build command
CMD="python3 src/examples/memory_manager_example.py --max-memory $MAX_MEMORY --page-size $PAGE_SIZE --threshold $THRESHOLD --log-level $LOG_LEVEL"

# Add boolean flags if enabled
if [ "$QUANTIZATION" = true ]; then
  CMD="$CMD --quantization"
fi

if [ "$OFFLOADING" = true ]; then
  CMD="$CMD --offloading"
fi

if [ "$FORCE_EVICTION" = true ]; then
  CMD="$CMD --force-eviction"
fi

# Display run configuration
echo "Running memory manager example with:"
echo "  Maximum memory: $MAX_MEMORY MB"
echo "  Page size: $PAGE_SIZE tokens"
echo "  Memory threshold: $THRESHOLD%"
echo "  Quantization: $QUANTIZATION"
echo "  Offloading: $OFFLOADING"
echo "  Force eviction: $FORCE_EVICTION"
echo "  Log level: $LOG_LEVEL"
echo ""

# Run the example
$CMD 