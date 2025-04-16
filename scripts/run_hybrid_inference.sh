#!/bin/bash
# Run Hybrid Inference Demonstration
#
# This script demonstrates the multi-pronged optimization approach for efficient
# LLM inference on mobile devices, combining:
# - Edge-Cloud Collaborative Inference
# - Runtime Token Pruning
# - Layer Compression and Skipping
# - Model-Aware Memory Management

set -e  # Exit on error

# Default settings
MODEL_NAME="gpt2"
PROMPT="Test memory compression, layer skipping, and token pruning with pruning action"
TOKENS=150 
DEVICE="auto"
LOW_MEM_BUDGET=5    # Budget used when forcing quantization
HIGH_MEM_BUDGET=10000 # Budget used to measure baseline memory without quantizing

# Benchmark comparison mode
COMPARISON_MODE="all" # Default comparison

DETAILED_LOG=false

# Help function
show_help() {
    echo "Usage: $0 [options] [--compare-mode]"
    echo
    echo "Runs comparison benchmarks using hybrid_inference.py"
    echo
    echo "General Options:"
    echo "  --model-name NAME     Hugging Face model name (default: gpt2)"
    echo "  --prompt TEXT         Input prompt (default: test sentence)"
    echo "  --tokens N            Number of tokens to generate (default: 150)"
    echo "  --device DEV          Device (cpu, cuda, auto) (default: auto)"
    # echo "  --low-budget MB       Low memory budget for forcing quantization (default: 5)" # Can add later if needed
    # echo "  --high-budget MB      High memory budget for baseline memory measurement (default: 10000)" # Can add later if needed
    echo "  --detailed            Enable detailed logging for both runs"
    echo "  --help                Show this help message and exit"
    echo
    echo "Comparison Modes (Choose one):"
    echo "  --compare-layer-skip          Compare Layer Skipping only vs Baseline (No Optimizations)"
    echo "  --compare-memory              Compare Memory Quantization (Low Budget) vs Memory Measurement (High Budget)"
    echo "  --compare-all                 Compare All Enabled (Layer Skip + Mem Quant Low Budget) vs Baseline (No Optimizations) [DEFAULT]"
    echo "  --compare-mem-overhead        Compare Memory Measurement (High Budget) vs Baseline (No Optimizations)"
    echo "  --compare-skip-plus-mem-overhead Compare Skip + Memory Measurement (High Budget) vs Baseline (No Optimizations)"
    echo "  --compare-token-pruning       [Experimental] Compare Token Pruning only vs Baseline (Requires functional Python code)"
    echo "  --compare-edge-cloud          Compare Edge-Cloud only vs Baseline (No Optimizations)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --tokens)
            TOKENS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        # --low-budget) LOW_MEM_BUDGET="$2"; shift 2 ;; # Example if budget args are needed
        # --high-budget) HIGH_MEM_BUDGET="$2"; shift 2 ;;
        --detailed)
            DETAILED_LOG=true
            shift
            ;;
        --compare-layer-skip)
            COMPARISON_MODE="layer_skip"
            shift
            ;;
        --compare-memory)
            COMPARISON_MODE="memory"
            shift
            ;;
        --compare-mem-overhead)
            COMPARISON_MODE="mem_overhead"
            shift
            ;;
        --compare-skip-plus-mem-overhead)
            COMPARISON_MODE="skip_plus_mem_overhead"
            shift
            ;;
        --compare-token-pruning)
            COMPARISON_MODE="token_pruning"
            shift
            ;;
        --compare-edge-cloud)
            COMPARISON_MODE="edge_cloud"
            shift
            ;;
        --compare-all)
            COMPARISON_MODE="all"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# --- Set up based on Comparison Mode ---
OPTIMIZED_FLAGS=""
BASELINE_FLAGS=""
COMPARISON_TITLE=""
JSON_COMPARISON_TYPE=""
OPTIMIZED_RUN_LABEL=""
BASELINE_RUN_LABEL=""
METRICS_TO_COMPARE=() # Array to hold metrics like time, mem, skips, quants

case $COMPARISON_MODE in
    layer_skip)
        OPTIMIZED_FLAGS="--layer-opt"
        BASELINE_FLAGS="" # No opt flags for baseline
        COMPARISON_TITLE="LAYER SKIP ONLY VS BASELINE"
        JSON_COMPARISON_TYPE="layer_skip_vs_baseline"
        OPTIMIZED_RUN_LABEL="Layer Skip Only"
        BASELINE_RUN_LABEL="Baseline (No Opts)"
        METRICS_TO_COMPARE=("time" "skips")
        ;;
    memory)
        OPTIMIZED_FLAGS="--memory-opt --mem-budget $LOW_MEM_BUDGET"
        BASELINE_FLAGS="--memory-opt --mem-budget $HIGH_MEM_BUDGET"
        COMPARISON_TITLE="MEMORY QUANTIZATION VS BASELINE MEMORY"
        JSON_COMPARISON_TYPE="memory_quantization_vs_baseline"
        OPTIMIZED_RUN_LABEL="Quantized (Low Budget)"
        BASELINE_RUN_LABEL="Unquantized (High Budget)"
        METRICS_TO_COMPARE=("time" "mem" "quants")
        ;;
    mem_overhead)
        OPTIMIZED_FLAGS="--memory-opt --mem-budget $HIGH_MEM_BUDGET"
        BASELINE_FLAGS="" # No opt flags for baseline
        COMPARISON_TITLE="MEMORY OVERHEAD VS BASELINE"
        JSON_COMPARISON_TYPE="mem_overhead_vs_baseline"
        OPTIMIZED_RUN_LABEL="Memory Measure (High Budget)"
        BASELINE_RUN_LABEL="Baseline (No Opts)"
        METRICS_TO_COMPARE=("time" "mem" "quants") # Compare mem even if expected to be same
        ;;
    skip_plus_mem_overhead)
        OPTIMIZED_FLAGS="--layer-opt --memory-opt --mem-budget $HIGH_MEM_BUDGET"
        BASELINE_FLAGS="" # No opt flags for baseline
        COMPARISON_TITLE="SKIP + MEMORY OVERHEAD VS BASELINE"
        JSON_COMPARISON_TYPE="skip_plus_mem_overhead_vs_baseline"
        OPTIMIZED_RUN_LABEL="Skip + Mem Measure (High Budget)"
        BASELINE_RUN_LABEL="Baseline (No Opts)"
        METRICS_TO_COMPARE=("time" "mem" "skips" "quants")
        ;;
    token_pruning)
        OPTIMIZED_FLAGS="--token-pruning"
        BASELINE_FLAGS=""
        COMPARISON_TITLE="[EXPERIMENTAL] TOKEN PRUNING VS BASELINE"
        JSON_COMPARISON_TYPE="token_pruning_vs_baseline"
        OPTIMIZED_RUN_LABEL="Token Pruning Only"
        BASELINE_RUN_LABEL="Baseline (No Opts)"
        METRICS_TO_COMPARE=("time" "pruned")
        echo "WARNING: --compare-token-pruning assumes pruning logic in Python is functional and uncommented."
        ;;
    edge_cloud)
        OPTIMIZED_FLAGS="--edge-cloud"
        BASELINE_FLAGS="" # No opt flags for baseline
        COMPARISON_TITLE="EDGE-CLOUD ONLY VS BASELINE"
        JSON_COMPARISON_TYPE="edge_cloud_vs_baseline"
        OPTIMIZED_RUN_LABEL="Edge-Cloud Only"
        BASELINE_RUN_LABEL="Baseline (No Opts)"
        METRICS_TO_COMPARE=("time" "offloaded") # Add offloaded metric
        ;;
    all | *)
        # Default to comparing all enabled vs baseline
        OPTIMIZED_FLAGS="--layer-opt --memory-opt --mem-budget $LOW_MEM_BUDGET --edge-cloud --token-pruning"
        BASELINE_FLAGS="--memory-opt --mem-budget $HIGH_MEM_BUDGET"
        COMPARISON_TITLE="ALL ENABLED (SKIP+QUANT+EDGE+PRUNE) VS BASELINE (MEM MEASURE ONLY)"
        JSON_COMPARISON_TYPE="all_vs_baseline_mem_measure_edge_prune"
        OPTIMIZED_RUN_LABEL="All Enabled (Skip+Quant+Edge+Prune)"
        BASELINE_RUN_LABEL="Baseline (Mem Measure Only)"
        METRICS_TO_COMPARE=("time" "mem" "skips" "quants" "offloaded" "pruned")
        COMPARISON_MODE="all" # Ensure mode is set for default case
        ;;
esac

# Common arguments
# MEM_BUDGET not needed here as flags include specific budgets
COMMON_ARGS="--model-name $MODEL_NAME --prompt \"$PROMPT\" --tokens $TOKENS --device $DEVICE"
if [ "$DETAILED_LOG" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --detailed"
fi

echo "Running benchmark comparison: $COMPARISON_TITLE"
timestamp=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p results

# Log files and JSON output file names
OPT_OUT_FILE="optimized_run_${COMPARISON_MODE}_${timestamp}.log"
BASE_OUT_FILE="baseline_run_${COMPARISON_MODE}_${timestamp}.log"
JSON_OUT_FILE="results/comparison_${COMPARISON_MODE}_${timestamp}.json"

# --- Run Optimized --- 
echo "1. Running OPTIMIZED ($OPTIMIZED_RUN_LABEL)..."
OPTIMIZED_CMD="python src/hybrid_inference.py $COMMON_ARGS $OPTIMIZED_FLAGS"
echo "   Command: $OPTIMIZED_CMD"
exec 3>&1 
OPTIMIZED_EXIT_CODE=0
eval "$OPTIMIZED_CMD" 2>&1 | tee "$OPT_OUT_FILE" >&3 || OPTIMIZED_EXIT_CODE=$?
exec 3>&-
if [ $OPTIMIZED_EXIT_CODE -ne 0 ]; then echo "ERROR: Optimized run failed... Check $OPT_OUT_FILE."; exit 1; fi
sync 
echo "   Finished Optimized Run."

# --- Run Baseline --- 
echo "2. Running BASELINE ($BASELINE_RUN_LABEL)..."
BASELINE_CMD="python src/hybrid_inference.py $COMMON_ARGS $BASELINE_FLAGS"
echo "   Command: $BASELINE_CMD"
exec 3>&1
BASELINE_EXIT_CODE=0
eval "$BASELINE_CMD" 2>&1 | tee "$BASE_OUT_FILE" >&3 || BASELINE_EXIT_CODE=$?
exec 3>&-
if [ $BASELINE_EXIT_CODE -ne 0 ]; then echo "ERROR: Baseline run failed... Check $BASE_OUT_FILE."; exit 1; fi
sync 
echo "   Finished Baseline Run."


# --- Extract Metrics --- 
echo ""
echo "Extracting metrics for $COMPARISON_MODE comparison..."

# Initialize metric variables with defaults
OPT_LOOP_TIME=0; OPT_PEAK_MEM=0; OPT_LAYER_SKIPS=0; OPT_QUANT_EVENTS=0; OPT_LAYERS_OFFLOADED=0; OPT_TOKENS_PRUNED=0
BASE_LOOP_TIME=0; BASE_PEAK_MEM=0; BASE_LAYER_SKIPS=0; BASE_QUANT_EVENTS=0; BASE_LAYERS_OFFLOADED=0; BASE_TOKENS_PRUNED=0

# Helper function for robust extraction
extract_metric() {
    local file="$1"
    local pattern="$2"
    local default="$3"

    # Find the last line matching pattern, remove prefix up to colon+space, remove suffix after number
    local result=$(grep "$pattern" "$file" | tail -n 1 | sed -e 's/^.*:[[:space:]]*//' -e 's/[^0-9.].*$//' || echo "$default")

    # Final safety check in bash
    if ! [[ "$result" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        # Add debug log if validation fails
        # echo "DEBUG: Validation failed for result='$result', pattern='$pattern', file='$file'" >&2
        result="$default"
    fi
    echo "$result"
}

OPT_LOOP_TIME=$(extract_metric "$OPT_OUT_FILE" "Generation Loop Time:" "0")
BASE_LOOP_TIME=$(extract_metric "$BASE_OUT_FILE" "Generation Loop Time:" "0")
echo "   Extracted OPT_LOOP_TIME: '$OPT_LOOP_TIME'"
echo "   Extracted BASE_LOOP_TIME: '$BASE_LOOP_TIME'"

# Extract other metrics only if needed for the comparison mode
if [[ " ${METRICS_TO_COMPARE[*]} " =~ " mem " ]]; then
    OPT_PEAK_MEM=$(extract_metric "$OPT_OUT_FILE" "Peak Memory Usage:" "0.0")
    BASE_PEAK_MEM=$(extract_metric "$BASE_OUT_FILE" "Peak Memory Usage:" "0.0")
    # Handle cases where memory manager wasn't enabled (grep finds nothing)
    if ! grep -q "Peak Memory Usage:" "$OPT_OUT_FILE"; then OPT_PEAK_MEM="0.0"; fi
    if ! grep -q "Peak Memory Usage:" "$BASE_OUT_FILE"; then BASE_PEAK_MEM="0.0"; fi
fi
echo "   Extracted OPT_PEAK_MEM: '$OPT_PEAK_MEM'"
echo "   Extracted BASE_PEAK_MEM: '$BASE_PEAK_MEM'"

if [[ " ${METRICS_TO_COMPARE[*]} " =~ " skips " ]]; then
    # Layers skipped: Need the integer value
    OPT_LAYER_SKIPS=$(extract_metric "$OPT_OUT_FILE" "Layers skipped (decision count):" "0")
    BASE_LAYER_SKIPS=$(extract_metric "$BASE_OUT_FILE" "Layers skipped (decision count):" "0")
fi
echo "   Extracted OPT_LAYER_SKIPS: '$OPT_LAYER_SKIPS'"
echo "   Extracted BASE_LAYER_SKIPS: '$BASE_LAYER_SKIPS'"

if [[ " ${METRICS_TO_COMPARE[*]} " =~ " quants " ]]; then
    # Quantization Events: Need the integer value
    OPT_QUANT_EVENTS=$(extract_metric "$OPT_OUT_FILE" "Quantization Events:" "0")
    BASE_QUANT_EVENTS=$(extract_metric "$BASE_OUT_FILE" "Quantization Events:" "0")
fi
echo "   Extracted OPT_QUANT_EVENTS: '$OPT_QUANT_EVENTS'"
echo "   Extracted BASE_QUANT_EVENTS: '$BASE_QUANT_EVENTS'"

if [[ " ${METRICS_TO_COMPARE[*]} " =~ " offloaded " ]]; then
    # Layers Offloaded: Need the integer value
    OPT_LAYERS_OFFLOADED=$(extract_metric "$OPT_OUT_FILE" "Layers Offloaded:" "0")
    BASE_LAYERS_OFFLOADED=$(extract_metric "$BASE_OUT_FILE" "Layers Offloaded:" "0")
fi
echo "   Extracted OPT_LAYERS_OFFLOADED: '$OPT_LAYERS_OFFLOADED'"
echo "   Extracted BASE_LAYERS_OFFLOADED: '$BASE_LAYERS_OFFLOADED'"

if [[ " ${METRICS_TO_COMPARE[*]} " =~ " pruned " ]]; then
    # Tokens Pruned: Need the integer value
    OPT_TOKENS_PRUNED=$(extract_metric "$OPT_OUT_FILE" "Tokens Pruned:" "0")
    BASE_TOKENS_PRUNED=$(extract_metric "$BASE_OUT_FILE" "Tokens Pruned:" "0")
fi
echo "   Extracted OPT_TOKENS_PRUNED: '$OPT_TOKENS_PRUNED'"
echo "   Extracted BASE_TOKENS_PRUNED: '$BASE_TOKENS_PRUNED'"

# --- Construct and Save JSON --- 
echo ""
echo "Saving results to $JSON_OUT_FILE ..."

JSON_CONTENT=$(cat <<EOF
{
  "timestamp": "$timestamp",
  "comparison_type": "$JSON_COMPARISON_TYPE",
  "model_name": "$MODEL_NAME",
  "prompt": "$PROMPT",
  "tokens": $TOKENS,
  "device": "$DEVICE",
  "optimized_run": {
    "label": "$OPTIMIZED_RUN_LABEL",
    "flags": "$OPTIMIZED_FLAGS",
    "loop_time_s": $OPT_LOOP_TIME,
    "peak_memory_mb": $OPT_PEAK_MEM,
    "layers_skipped": $OPT_LAYER_SKIPS,
    "quant_events": $OPT_QUANT_EVENTS,
    "layers_offloaded": $OPT_LAYERS_OFFLOADED,
    "tokens_pruned": $OPT_TOKENS_PRUNED
  },
  "baseline_run": {
    "label": "$BASELINE_RUN_LABEL",
    "flags": "$BASELINE_FLAGS",
    "loop_time_s": $BASE_LOOP_TIME,
    "peak_memory_mb": $BASE_PEAK_MEM,
    "layers_skipped": $BASE_LAYER_SKIPS,
    "quant_events": $BASE_QUANT_EVENTS,
    "layers_offloaded": $BASE_LAYERS_OFFLOADED,
    "tokens_pruned": $BASE_TOKENS_PRUNED
  }
}
EOF
)

echo "$JSON_CONTENT" | jq '.' > "$JSON_OUT_FILE" # Use jq for basic validation/pretty-print if available
if [ $? -ne 0 ]; then echo "ERROR: Failed to write JSON (maybe install jq?). Saving raw."; echo "$JSON_CONTENT" > "$JSON_OUT_FILE"; fi
echo "   Successfully saved comparison results."


# --- Display Comparison Table (Console Output) --- 
echo ""
echo "==================================================="
echo "BENCHMARK COMPARISON - $COMPARISON_TITLE"
echo "Prompt: $PROMPT"
echo "Tokens: $TOKENS | Model: $MODEL_NAME | Device: $DEVICE"
echo "==================================================="
# Dynamically build header and rows based on METRICS_TO_COMPARE
HEADER_FMT="%-25s | %-25s | %-25s\n"
ROW_FMT_S="%-25s | %-25s | %-25s\n"  # Format for strings/integers
ROW_FMT_F="%-25s | %-25.2f | %-25.2f\n" # Format for floats
SEPARATOR=$(printf -- '-%.0s' {1..81})

printf "$HEADER_FMT" "Metric" "$OPTIMIZED_RUN_LABEL" "$BASELINE_RUN_LABEL"
echo "$SEPARATOR"

if [[ " ${METRICS_TO_COMPARE[*]} " =~ " time " ]]; then
    printf "$ROW_FMT_F" "Loop Time (s)" "$OPT_LOOP_TIME" "$BASE_LOOP_TIME"
fi
if [[ " ${METRICS_TO_COMPARE[*]} " =~ " mem " ]]; then
    printf "$ROW_FMT_F" "Peak Memory (MB)" "$OPT_PEAK_MEM" "$BASE_PEAK_MEM"
fi
if [[ " ${METRICS_TO_COMPARE[*]} " =~ " skips " ]]; then
    printf "$ROW_FMT_S" "Layers Skipped" "$OPT_LAYER_SKIPS" "$BASE_LAYER_SKIPS"
fi
if [[ " ${METRICS_TO_COMPARE[*]} " =~ " quants " ]]; then
    printf "$ROW_FMT_S" "Quantization Events" "$OPT_QUANT_EVENTS" "$BASE_QUANT_EVENTS"
fi
if [[ " ${METRICS_TO_COMPARE[*]} " =~ " offloaded " ]]; then
    printf "$ROW_FMT_S" "Layers Offloaded" "$OPT_LAYERS_OFFLOADED" "$BASE_LAYERS_OFFLOADED"
fi
if [[ " ${METRICS_TO_COMPARE[*]} " =~ " pruned " ]]; then
    printf "$ROW_FMT_S" "Tokens Pruned" "$OPT_TOKENS_PRUNED" "$BASE_TOKENS_PRUNED"
fi
echo "$SEPARATOR"

# Calculate differences (requires bc or similar)
echo "Calculation Hints (requires 'bc'):"
if command -v bc >/dev/null 2>&1; then
    if [[ " ${METRICS_TO_COMPARE[*]} " =~ " time " ]]; then
        TIME_SAVED=$(echo "scale=2; $BASE_LOOP_TIME - $OPT_LOOP_TIME" | bc)
        echo "Time Saved (Baseline - Opt): $TIME_SAVED s"
        if (( $(echo "$OPT_LOOP_TIME > 0" | bc -l) )); then
            SPEEDUP=$(echo "scale=2; $BASE_LOOP_TIME / $OPT_LOOP_TIME" | bc)
            echo "Speedup: ${SPEEDUP}x"
        fi
    fi
    if [[ " ${METRICS_TO_COMPARE[*]} " =~ " mem " ]]; then
         MEM_SAVED=$(echo "scale=2; $BASE_PEAK_MEM - $OPT_PEAK_MEM" | bc)
         echo "Memory Saved (Base - Opt): $MEM_SAVED MB"
    fi
else
    echo " 'bc' command not found. Cannot calculate differences automatically."
fi
echo "==================================================="

# --- Extract and Display Generated  ---
echo ""
echo "--- Generated Text Comparison ---"

# Extract text (handle potential missing lines)
OPT_GENERATED_TEXT=$(grep 'INFO:hybrid_inference:Newly Generated Text:' "$OPT_OUT_FILE" | tail -n 1 | sed -e 's/^.*INFO:hybrid_inference:Newly Generated Text: //' -e 's/^[[:space:].]*//' || echo "[Generated text not found in log]")
BASE_GENERATED_TEXT=$(grep 'INFO:hybrid_inference:Newly Generated Text:' "$BASE_OUT_FILE" | tail -n 1 | sed -e 's/^.*INFO:hybrid_inference:Newly Generated Text: //' -e 's/^[[:space:].]*//' || echo "[Generated text not found in log]")

printf "%-25s: %s\n" "Optimized ($OPTIMIZED_RUN_LABEL)" "$OPT_GENERATED_TEXT"
printf "%-25s: %s\n" "Baseline ($BASELINE_RUN_LABEL)" "$BASE_GENERATED_TEXT"
echo "---------------------------------"
# --- <<< END: Generated Text Comparison ---

# Clean up log files
# Keep logs for inspection
# rm "$OPT_OUT_FILE" "$BASE_OUT_FILE"

echo "Benchmark complete! JSON results saved to $JSON_OUT_FILE" 