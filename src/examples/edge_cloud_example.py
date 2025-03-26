"""
Example of Edge-Cloud Collaborative Inference

This script demonstrates how to use the Edge-Cloud Manager with all components:
- Device monitoring
- Cloud communication
- Privacy protection
- Mini-LLM handling

It shows a complete example of running LLM inference with dynamic layer partitioning.
"""

import os
import sys
import torch
import numpy as np
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_cloud.edge_cloud_manager import EdgeCloudManager
from edge_cloud.device_monitor import DeviceMonitor
from edge_cloud.cloud_client import MockCloudClient
from edge_cloud.mini_llm import MiniLLMHandler, TokenDifficultyEstimator
from edge_cloud.security import PrivacyProtection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("edge_cloud_example")


def on_metrics_update(metrics):
    """Callback for when device metrics are updated."""
    logger.info(f"Network bandwidth: {metrics['network']['bandwidth_mbps']:.2f} Mbps")
    logger.info(f"CPU usage: {metrics['hardware']['cpu_usage_percent']:.1f}%")


def run_example():
    """Run a complete example of edge-cloud collaborative inference."""
    logger.info("Starting Edge-Cloud Collaborative Inference example")
    
    # Get configuration from environment variables or use defaults
    model_name = os.environ.get("MODEL_NAME", "distilgpt2")
    prompt = os.environ.get("INPUT_PROMPT", "Once upon a time in a land far away,")
    num_tokens = int(os.environ.get("NUM_TOKENS", "20"))
    energy_weight = float(os.environ.get("ENERGY_WEIGHT", "0.3"))
    latency_weight = float(os.environ.get("LATENCY_WEIGHT", "0.4"))
    memory_weight = float(os.environ.get("MEMORY_WEIGHT", "0.3"))
    cloud_latency = float(os.environ.get("CLOUD_LATENCY", "0.5"))
    encryption_enabled = os.environ.get("ENCRYPTION_ENABLED", "True").lower() == "true"
    force_local = os.environ.get("FORCE_LOCAL", "False").lower() == "true"
    force_cloud = os.environ.get("FORCE_CLOUD", "False").lower() == "true"
    
    # 1. Load models - in a real scenario, we'd have a small and large model
    logger.info(f"Loading model: {model_name}...")
    
    # Use a tiny model for both main and mini-LLM in this example
    # In a real scenario, main_model would be larger and mini_llm smaller
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        main_model = AutoModelForCausalLM.from_pretrained(model_name)
        mini_model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        logger.info("Falling back to distilgpt2")
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        main_model = AutoModelForCausalLM.from_pretrained(model_name)
        mini_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set to evaluation mode
    main_model.eval()
    mini_model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_model.to(device)
    mini_model.to(device)
    
    logger.info(f"Models loaded and moved to {device}")
    
    # 2. Set up device monitoring
    logger.info("Setting up device monitoring...")
    device_monitor = DeviceMonitor(
        monitoring_interval=10.0,
        background_monitoring=True,
        callback=on_metrics_update
    )
    
    # 3. Set up cloud client (using mock client for this example)
    logger.info(f"Setting up cloud client with latency {cloud_latency}s...")
    cloud_client = MockCloudClient(
        latency=cloud_latency,  # Simulated cloud latency
        failure_rate=0.1  # 10% chance of request failure for testing resilience
    )
    
    # 4. Set up mini-LLM handler
    logger.info("Setting up mini-LLM handler...")
    difficulty_estimator = TokenDifficultyEstimator(
        difficulty_threshold=0.7  # Tokens with difficulty > 0.7 are "hard"
    )
    
    mini_llm_handler = MiniLLMHandler(
        model=mini_model,
        tokenizer=tokenizer,
        difficulty_estimator=difficulty_estimator,
        confidence_threshold=0.8
    )
    
    # 5. Set up privacy protection
    logger.info(f"Setting up privacy protection (encryption: {encryption_enabled})...")
    privacy_protection = PrivacyProtection(
        encryption_enabled=encryption_enabled,
        reduction_enabled=True,
        reduction_ratio=0.7  # Reduce dimensions to 70% of original
    )
    
    # 6. Create the Edge-Cloud Manager
    logger.info(f"Creating Edge-Cloud Manager with weights: energy={energy_weight}, latency={latency_weight}, memory={memory_weight}...")
    edge_cloud_manager = EdgeCloudManager(
        model=main_model,
        device_monitor=device_monitor,
        cloud_client=cloud_client,
        mini_llm=mini_llm_handler,
        privacy_protection=privacy_protection,
        energy_weight=energy_weight,
        latency_weight=latency_weight,
        memory_weight=memory_weight
    )
    
    # Set forced execution modes if specified
    if force_local:
        logger.info("Forcing all computation to be local")
        edge_cloud_manager.force_execution_mode = "local"
    elif force_cloud:
        logger.info("Forcing all computation to be in the cloud")
        edge_cloud_manager.force_execution_mode = "remote"
    
    # 7. Run inference with the provided prompt
    logger.info("Running inference...")
    logger.info(f"Input: {prompt}")
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Run with different approaches for comparison
    
    # Local-only inference
    logger.info("Running local-only inference...")
    start_time = time.time()
    local_output = edge_cloud_manager.process_locally(input_ids)
    local_time = time.time() - start_time
    
    # Hybrid inference with dynamic partitioning
    logger.info("Running hybrid inference with dynamic partitioning...")
    start_time = time.time()
    hybrid_output = edge_cloud_manager.process_hybrid(input_ids)
    hybrid_time = time.time() - start_time
    
    # Generate tokens
    logger.info(f"Generating {num_tokens} tokens with hybrid approach...")
    generated_prompt = input_ids
    generated_text = prompt
    
    for i in range(num_tokens):  # Generate tokens
        logger.info(f"Generating token {i+1}/{num_tokens}...")
        
        # Check current conditions
        metrics = device_monitor.get_metrics()
        logger.info(f"Current bandwidth: {metrics['network']['bandwidth_mbps']:.2f} Mbps")
        
        # Determine if we should use mini-LLM for this token
        should_use_mini = mini_llm_handler.should_use_mini_llm(generated_prompt, None)
        
        if should_use_mini:
            logger.info("Using mini-LLM for this token (deemed 'easy')")
            with torch.no_grad():
                outputs = mini_model(generated_prompt)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        else:
            logger.info("Using hybrid approach for this token (deemed 'hard')")
            # Use the hybrid processing approach
            outputs = edge_cloud_manager.process_hybrid(generated_prompt)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Update the prompt with the new token
        generated_prompt = torch.cat([generated_prompt, next_token_id], dim=1)
        
        # Convert to text
        new_text = tokenizer.decode(next_token_id[0])
        generated_text += new_text
        logger.info(f"Generated token: '{new_text}'")
        
    logger.info(f"\nFinal generated text: {generated_text}")
    
    # 8. Print performance comparison
    logger.info("\nPerformance Comparison:")
    logger.info(f"Local-only inference time: {local_time:.4f} seconds")
    logger.info(f"Hybrid inference time: {hybrid_time:.4f} seconds")
    speedup = local_time / hybrid_time if hybrid_time > 0 else 0
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # In a real scenario, hybrid would often be faster due to cloud acceleration
    # In our mock example, the overhead might make hybrid slower
    
    # 9. Print Edge-Cloud Manager metrics
    logger.info("\nEdge-Cloud Manager Metrics:")
    logger.info(f"Cloud requests: {edge_cloud_manager.metrics['cloud_requests']}")
    logger.info(f"Local inferences: {edge_cloud_manager.metrics['local_inferences']}")
    logger.info(f"Mini-LLM usage: {edge_cloud_manager.metrics['mini_llm_usage']}")
    total_tokens = edge_cloud_manager.metrics['total_tokens']
    if total_tokens > 0:
        mini_llm_percentage = (edge_cloud_manager.metrics['mini_llm_usage'] / total_tokens) * 100
        logger.info(f"Mini-LLM usage percentage: {mini_llm_percentage:.1f}%")
    
    # 10. Clean up
    logger.info("\nCleaning up...")
    device_monitor.stop_background_monitoring()
    cloud_client.close()
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    run_example() 