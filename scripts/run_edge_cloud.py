#!/usr/bin/env python
"""
Run Edge-Cloud Collaborative Inference Example

This script demonstrates the Edge-Cloud collaborative inference approach
for efficient LLM inference on mobile devices by dynamically partitioning
layers between the device and cloud.

Usage:
    python scripts/run_edge_cloud.py [--model MODEL] [--prompt PROMPT] [--tokens NUM_TOKENS]
"""

import os
import sys
import argparse
import logging
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import necessary modules
from examples.edge_cloud_example import run_example

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run Edge-Cloud Collaborative Inference Example")
    
    parser.add_argument("--model", type=str, default="distilgpt2", 
                        help="Model name to use (default: distilgpt2)")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land far away,", 
                        help="Prompt to start generation with")
    parser.add_argument("--tokens", type=int, default=20, 
                        help="Number of tokens to generate (default: 20)")
    parser.add_argument("--energy-weight", type=float, default=0.3, 
                        help="Weight of energy in cost function (default: 0.3)")
    parser.add_argument("--latency-weight", type=float, default=0.4, 
                        help="Weight of latency in cost function (default: 0.4)")
    parser.add_argument("--memory-weight", type=float, default=0.3, 
                        help="Weight of memory in cost function (default: 0.3)")
    parser.add_argument("--cloud-latency", type=float, default=0.5, 
                        help="Simulated cloud latency in seconds (default: 0.5)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--no-encryption", action="store_true",
                        help="Disable encryption for cloud communication")
    parser.add_argument("--force-local", action="store_true",
                        help="Force all computation to be local")
    parser.add_argument("--force-cloud", action="store_true",
                        help="Force all computation to be in the cloud")
    
    return parser.parse_args()

def main():
    """Main entry point for the script"""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("edge_cloud_script")
    logger.info("Starting Edge-Cloud Collaborative Inference script")
    
    # Print arguments
    logger.info(f"Using model: {args.model}")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Generating {args.tokens} tokens")
    logger.info(f"Energy/Latency/Memory weights: {args.energy_weight}/{args.latency_weight}/{args.memory_weight}")
    logger.info(f"Simulated cloud latency: {args.cloud_latency}s")
    logger.info(f"Encryption enabled: {not args.no_encryption}")
    
    if args.force_local and args.force_cloud:
        logger.error("Cannot force both local and cloud computation. Choose one.")
        return
    
    # Set environment variables to control the example behavior
    os.environ["MODEL_NAME"] = args.model
    os.environ["INPUT_PROMPT"] = args.prompt
    os.environ["NUM_TOKENS"] = str(args.tokens)
    os.environ["ENERGY_WEIGHT"] = str(args.energy_weight)
    os.environ["LATENCY_WEIGHT"] = str(args.latency_weight)
    os.environ["MEMORY_WEIGHT"] = str(args.memory_weight)
    os.environ["CLOUD_LATENCY"] = str(args.cloud_latency)
    os.environ["ENCRYPTION_ENABLED"] = str(not args.no_encryption)
    os.environ["FORCE_LOCAL"] = str(args.force_local)
    os.environ["FORCE_CLOUD"] = str(args.force_cloud)
    
    # Run the example
    start_time = time.time()
    try:
        run_example()
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error running example: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 