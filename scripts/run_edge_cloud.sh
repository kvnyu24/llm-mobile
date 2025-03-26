#!/bin/bash

# Script to run the Edge-Cloud Collaborative Inference example
# with configurable parameters

# Default values
MODEL_NAME="distilgpt2"
INPUT_PROMPT="Once upon a time in a land far away,"
NUM_TOKENS=20
ENERGY_WEIGHT=0.3
LATENCY_WEIGHT=0.4
MEMORY_WEIGHT=0.3
CLOUD_LATENCY=0.5
ENCRYPTION_ENABLED="True"
FORCE_LOCAL="False"
FORCE_CLOUD="False"
LOG_LEVEL="INFO"

# Show usage information
function show_usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --model <name>          Model name (default: distilgpt2)"
  echo "  --prompt <text>         Input prompt (default: 'Once upon a time in a land far away,')"
  echo "  --tokens <number>       Number of tokens to generate (default: 20)"
  echo "  --energy-weight <float> Weight for energy in cost function (default: 0.3)"
  echo "  --latency-weight <float> Weight for latency in cost function (default: 0.4)"
  echo "  --memory-weight <float> Weight for memory in cost function (default: 0.3)"
  echo "  --cloud-latency <float> Simulated cloud latency in seconds (default: 0.5)"
  echo "  --log-level <level>     Logging level (default: INFO)"
  echo "  --no-encryption         Disable encryption for cloud communication"
  echo "  --force-local           Force all computation to be local"
  echo "  --force-cloud           Force all computation to be in the cloud"
  echo "  --help                  Show this help message"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --prompt)
      INPUT_PROMPT="$2"
      shift 2
      ;;
    --tokens)
      NUM_TOKENS="$2"
      shift 2
      ;;
    --energy-weight)
      ENERGY_WEIGHT="$2"
      shift 2
      ;;
    --latency-weight)
      LATENCY_WEIGHT="$2"
      shift 2
      ;;
    --memory-weight)
      MEMORY_WEIGHT="$2"
      shift 2
      ;;
    --cloud-latency)
      CLOUD_LATENCY="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --no-encryption)
      ENCRYPTION_ENABLED="False"
      shift
      ;;
    --force-local)
      FORCE_LOCAL="True"
      FORCE_CLOUD="False"
      shift
      ;;
    --force-cloud)
      FORCE_CLOUD="True"
      FORCE_LOCAL="False"
      shift
      ;;
    --help)
      show_usage
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      ;;
  esac
done

# Check for conflicting flags
if [[ "$FORCE_LOCAL" == "True" && "$FORCE_CLOUD" == "True" ]]; then
  echo "Error: Cannot use both --force-local and --force-cloud at the same time"
  exit 1
fi

# Print the configuration
echo "Running Edge-Cloud Collaborative Inference with:"
echo "- Model: $MODEL_NAME"
echo "- Prompt: \"$INPUT_PROMPT\""
echo "- Tokens to generate: $NUM_TOKENS"
echo "- Energy weight: $ENERGY_WEIGHT"
echo "- Latency weight: $LATENCY_WEIGHT"
echo "- Memory weight: $MEMORY_WEIGHT"
echo "- Cloud latency: ${CLOUD_LATENCY}s"
echo "- Encryption: $ENCRYPTION_ENABLED"
echo "- Force local: $FORCE_LOCAL"
echo "- Force cloud: $FORCE_CLOUD"
echo "- Log level: $LOG_LEVEL"
echo

# Set environment variables for the example
export MODEL_NAME
export INPUT_PROMPT
export NUM_TOKENS
export ENERGY_WEIGHT
export LATENCY_WEIGHT
export MEMORY_WEIGHT
export CLOUD_LATENCY
export ENCRYPTION_ENABLED
export FORCE_LOCAL
export FORCE_CLOUD

# Set Python logging level
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOGLEVEL=$LOG_LEVEL

# Run the example
echo "Starting example..."
python src/examples/edge_cloud_example.py

# Clean up environment variables
unset MODEL_NAME
unset INPUT_PROMPT
unset NUM_TOKENS
unset ENERGY_WEIGHT
unset LATENCY_WEIGHT
unset MEMORY_WEIGHT
unset CLOUD_LATENCY
unset ENCRYPTION_ENABLED
unset FORCE_LOCAL
unset FORCE_CLOUD
unset LOGLEVEL 