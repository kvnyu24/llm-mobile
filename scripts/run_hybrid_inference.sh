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
TOKENS=10
SEQ_LEN=8
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
DISABLE_TOKEN_PRUNING=false
DISABLE_LAYER_OPT=false
DISABLE_MEMORY_OPT=false
DISABLE_EDGE_CLOUD=false
DETAILED_LOG=false
SAVE_STATS=false
OUTPUT_FILE=""

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Demonstrates the multi-pronged optimization approach for efficient LLM inference"
    echo
    echo "Options:"
    echo "  --tokens N              Number of tokens to generate (default: 10)"
    echo "  --seq-len N             Initial sequence length (default: 8)"
    echo "  --hidden-size N         Hidden size dimension (default: 768)"
    echo "  --num-layers N          Number of transformer layers (default: 12)"
    echo "  --num-heads N           Number of attention heads (default: 12)"
    echo "  --no-token-pruning      Disable token pruning optimization"
    echo "  --no-layer-opt          Disable layer compression and skipping"
    echo "  --no-memory-opt         Disable memory management optimization"
    echo "  --no-edge-cloud         Disable edge-cloud partitioning"
    echo "  --detailed              Enable detailed logging"
    echo "  --save-stats            Save statistics to a JSON file"
    echo "  --output FILE           Output file for statistics"
    echo "  --benchmark             Run in benchmark mode (all optimizations on/off)"
    echo "  --help                  Show this help message and exit"
    echo
    echo "Examples:"
    echo "  $0 --tokens 20 --detailed"
    echo "  $0 --no-token-pruning --no-edge-cloud --save-stats"
    echo "  $0 --benchmark"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokens)
            TOKENS="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --no-token-pruning)
            DISABLE_TOKEN_PRUNING=true
            shift
            ;;
        --no-layer-opt)
            DISABLE_LAYER_OPT=true
            shift
            ;;
        --no-memory-opt)
            DISABLE_MEMORY_OPT=true
            shift
            ;;
        --no-edge-cloud)
            DISABLE_EDGE_CLOUD=true
            shift
            ;;
        --detailed)
            DETAILED_LOG=true
            shift
            ;;
        --save-stats)
            SAVE_STATS=true
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --benchmark)
            echo "Running benchmark mode..."
            timestamp=$(date +%Y%m%d_%H%M%S)
            
            # Run with all optimizations enabled
            echo "1. Running with all optimizations enabled..."
            python src/hybrid_inference.py --tokens "$TOKENS" --seq-len "$SEQ_LEN" --save-stats --output "hybrid_inference_all_enabled_${timestamp}.json"
            
            # Run with all optimizations disabled
            echo "2. Running with all optimizations disabled..."
            python src/hybrid_inference.py --tokens "$TOKENS" --seq-len "$SEQ_LEN" --no-token-pruning --no-layer-opt --no-memory-opt --no-edge-cloud --save-stats --output "hybrid_inference_all_disabled_${timestamp}.json"
            
            # Compare the results - create a comparison summary
            echo ""
            echo "==================================================="
            echo "BENCHMARK COMPARISON - OPTIMIZED VS UNOPTIMIZED"
            echo "==================================================="
            
            # Use jq to generate a detailed comparison
            if command -v jq >/dev/null 2>&1; then
                # Create a combined JSON file with both results
                jq -s '{optimized: .[0], unoptimized: .[1]}' "hybrid_inference_all_enabled_${timestamp}.json" "hybrid_inference_all_disabled_${timestamp}.json" > "hybrid_inference_comparison_${timestamp}.json"
                
                # Extract computation statistics
                opt_comp_time=$(jq -r '.optimized.computation_time.avg_computation_time_seconds' "hybrid_inference_comparison_${timestamp}.json")
                unopt_comp_time=$(jq -r '.unoptimized.computation_time.avg_computation_time_seconds' "hybrid_inference_comparison_${timestamp}.json")
                
                # Calculate speedup
                if (( $(echo "$unopt_comp_time > 0" | bc -l) )); then
                    speedup=$(echo "scale=2; $unopt_comp_time / $opt_comp_time" | bc)
                    speedup_percent=$(echo "scale=1; ($speedup - 1) * 100" | bc)
                else
                    speedup="N/A"
                    speedup_percent="N/A"
                fi
                
                # Display comparison
                echo "Computation Time Comparison:"
                echo "  Optimized:   ${opt_comp_time}s per token"
                echo "  Unoptimized: ${unopt_comp_time}s per token"
                echo "  Speedup:     ${speedup}x (${speedup_percent}%)"
                echo ""
                
                # Memory usage comparison
                opt_mem_saved=$(jq -r '.optimized.memory_saved_mb' "hybrid_inference_comparison_${timestamp}.json")
                unopt_mem_saved=$(jq -r '.unoptimized.memory_saved_mb' "hybrid_inference_comparison_${timestamp}.json")
                mem_diff=$(echo "scale=2; $opt_mem_saved - $unopt_mem_saved" | bc)
                
                echo "Memory Usage Comparison:"
                echo "  Optimized memory saved:   ${opt_mem_saved} MB"
                echo "  Unoptimized memory saved: ${unopt_mem_saved} MB"
                echo "  Difference:               ${mem_diff} MB"
                echo ""
                
                # Energy savings
                opt_energy=$(jq -r '.optimized.estimated_energy_savings_percent' "hybrid_inference_comparison_${timestamp}.json")
                unopt_energy=$(jq -r '.unoptimized.estimated_energy_savings_percent' "hybrid_inference_comparison_${timestamp}.json")
                energy_diff=$(echo "scale=1; $opt_energy - $unopt_energy" | bc)
                
                echo "Energy Savings Comparison:"
                echo "  Optimized energy savings:   ${opt_energy}%"
                echo "  Unoptimized energy savings: ${unopt_energy}%"
                echo "  Difference:                 ${energy_diff}%"
                echo ""
                
                # Overhead comparison
                opt_overhead=$(jq -r '.optimized.computation_time.overhead_time_seconds' "hybrid_inference_comparison_${timestamp}.json")
                unopt_overhead=$(jq -r '.unoptimized.computation_time.overhead_time_seconds' "hybrid_inference_comparison_${timestamp}.json")
                
                echo "Setup Overhead Comparison:"
                echo "  Optimized overhead:   ${opt_overhead}s per token"
                echo "  Unoptimized overhead: ${unopt_overhead}s per token"
                echo ""
                
                echo "Detailed comparison saved to: hybrid_inference_comparison_${timestamp}.json"
            else
                echo "jq command not found. Please install jq to see detailed comparisons."
                echo "Basic results saved to individual JSON files."
            fi
            
            echo "==================================================="
            echo "Benchmark complete!"
            echo "==================================================="
            
            exit 0
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

# Build the command with appropriate flags
CMD="python src/hybrid_inference.py --tokens $TOKENS --seq-len $SEQ_LEN --hidden-size $HIDDEN_SIZE --num-layers $NUM_LAYERS --num-heads $NUM_HEADS"

# Add optional flags based on user choices
if [ "$DISABLE_TOKEN_PRUNING" = true ]; then
    CMD="$CMD --no-token-pruning"
fi

if [ "$DISABLE_LAYER_OPT" = true ]; then
    CMD="$CMD --no-layer-opt"
fi

if [ "$DISABLE_MEMORY_OPT" = true ]; then
    CMD="$CMD --no-memory-opt"
fi

if [ "$DISABLE_EDGE_CLOUD" = true ]; then
    CMD="$CMD --no-edge-cloud"
fi

if [ "$DETAILED_LOG" = true ]; then
    CMD="$CMD --detailed"
fi

if [ "$SAVE_STATS" = true ]; then
    CMD="$CMD --save-stats"
    
    if [ -n "$OUTPUT_FILE" ]; then
        CMD="$CMD --output $OUTPUT_FILE"
    fi
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Display configuration
echo "============================================"
echo "HYBRID INFERENCE DEMONSTRATION"
echo "============================================"
echo "Configuration:"
echo " - Tokens to generate: $TOKENS"
echo " - Initial sequence length: $SEQ_LEN"
echo " - Model dimensions: ${HIDDEN_SIZE}d, ${NUM_LAYERS} layers, ${NUM_HEADS} heads"
echo " - Token pruning: $([ "$DISABLE_TOKEN_PRUNING" = true ] && echo "disabled" || echo "enabled")"
echo " - Layer optimization: $([ "$DISABLE_LAYER_OPT" = true ] && echo "disabled" || echo "enabled")"
echo " - Memory optimization: $([ "$DISABLE_MEMORY_OPT" = true ] && echo "disabled" || echo "enabled")"
echo " - Edge-Cloud partitioning: $([ "$DISABLE_EDGE_CLOUD" = true ] && echo "disabled" || echo "enabled")"
echo "============================================"

# Run the command
echo "Running: $CMD"
eval "$CMD"

echo "Hybrid inference complete!" 