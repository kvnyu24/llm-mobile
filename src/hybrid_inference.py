"""
Hybrid Inference for Transformer-based LLMs on Mobile Devices

This module demonstrates a multi-pronged approach to efficient LLM inference
on mobile devices, as discussed in research on efficient inference techniques.
It combines four key optimization techniques:

1. Edge-Cloud Collaborative Inference: Dynamic partitioning of model layers
   between device and cloud based on network conditions and computational costs.

2. Runtime Token Pruning: Dynamically removing less important tokens during
   inference to reduce computational costs of attention operations.

3. Layer Compression and Skipping: Selectively compressing or skipping certain
   transformer layers based on their importance for the current inference task.

4. Model-Aware Memory Management: Efficient management of key-value cache using
   paging, quantization, and eviction strategies.

Together, these approaches enable efficient inference of large language models
on resource-constrained mobile devices.
"""

import logging
import numpy as np
import random
import torch
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import torch.nn.functional as F

# Import the four key modules
from edge_cloud.edge_cloud_manager import EdgeCloudManager
from token_pruning.token_pruner import TokenPruner
from layer_compression.layer_compression_skipping import LayerCompressionAndSkipping
from memory_manager.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_inference")

def create_fake_hidden_state(batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    """Create a simulated hidden state tensor for demonstration purposes."""
    return torch.rand(batch_size, seq_len, hidden_size)

def create_fake_hidden_states(batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    """Create simulated hidden states for token pruning demonstration."""
    return torch.randn(batch_size, seq_len, hidden_size)

def create_fake_attention_matrix(seq_len):
    """Create a simulated attention matrix for token pruning demonstration."""
    # Create base attention pattern [seq_len, seq_len]
    attention_matrix = torch.zeros((seq_len, seq_len))
    
    # Create attention pattern where:
    # - First few tokens get medium attention (0.5)
    # - Middle tokens get medium-high attention (0.7)
    # - Recent tokens get high attention (0.85)
    # - Latest token gets very high attention (0.95)
    # - Self-attention is always high (0.9)
    
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:  # Self attention
                attention_matrix[i,j] = 0.9
            elif j == seq_len - 1:  # Latest token
                attention_matrix[i,j] = 0.95
            elif j < 3:  # First few tokens
                attention_matrix[i,j] = 0.5
            elif j > seq_len - 4:  # Recent tokens
                attention_matrix[i,j] = 0.85
            else:  # Middle tokens
                attention_matrix[i,j] = 0.7
                
    # Normalize each row
    attention_matrix = F.softmax(attention_matrix * 5.0, dim=-1)  # Lower temperature for less extreme distribution
    
    # Expand to [batch=1, heads=12, seq_len, seq_len]
    attention_matrix = attention_matrix.unsqueeze(0).unsqueeze(0)
    attention_matrix = attention_matrix.expand(1, 12, seq_len, seq_len)
    
    return attention_matrix

def simulate_network_conditions() -> Dict[str, float]:
    """Simulate varying network conditions for edge-cloud decisions."""
    # Returns bandwidth in Mbps and latency in ms
    return {
        "bandwidth_mbps": random.uniform(0.5, 10.0),
        "latency_ms": random.uniform(50, 500),
        "connected": random.random() > 0.1  # 10% chance of no connection
    }

def simulate_device_load() -> Dict[str, Any]:
    """Simulate device computational load for edge-cloud decisions."""
    return {
        "cpu_usage_percent": random.uniform(10, 90),
        "memory": {
            "available_mb": random.uniform(50, 500),
            "used_percent": random.uniform(30, 80)
        },
        "battery_level_percent": random.uniform(20, 100)
    }

def estimate_latency_savings(
    base_latency: float, 
    tokens_pruned: int, 
    layers_skipped: int, 
    compressed_layers: int,
    seq_len: int,
    num_layers: int
) -> float:
    """
    Estimate the latency savings from our optimizations.
    
    Args:
        base_latency: Base latency for processing a token through all layers
        tokens_pruned: Number of tokens pruned
        layers_skipped: Number of layers skipped
        compressed_layers: Number of compressed layers
        seq_len: Current sequence length
        num_layers: Number of layers in the model
        
    Returns:
        Estimated latency savings in ms
    """
    # Token pruning savings (quadratic effect on attention operations)
    token_pruning_factor = 1.0
    if seq_len > 0:
        token_reduction_ratio = tokens_pruned / seq_len
        # Quadratic effect due to attention matrix size reduction
        token_pruning_factor = (1.0 - token_reduction_ratio) ** 2
    
    # Layer skipping savings (linear with number of layers)
    layer_skip_factor = 1.0 - (layers_skipped / num_layers)
    
    # Compression savings (smaller effect than skipping)
    compression_factor = 1.0 - (0.3 * compressed_layers / num_layers)
    
    # Combined effect (multiplicative)
    total_factor = token_pruning_factor * layer_skip_factor * compression_factor
    
    # Calculate savings
    optimized_latency = base_latency * total_factor
    savings = base_latency - optimized_latency
    
    return savings

def run_inference(
    input_tokens: torch.Tensor, 
    max_new_tokens: int = 20, 
    model_config: Optional[Dict[str, Any]] = None,
    enable_token_pruning: bool = True,
    enable_layer_optimization: bool = True,
    enable_memory_management: bool = True,
    enable_edge_cloud: bool = True,
    detailed_logging: bool = True,
    simulated_base_latency_ms: float = 50.0  # Fake base latency per token
) -> Dict[str, Any]:
    """
    Run a simulated inference process using the multi-pronged optimization approach.
    
    This function demonstrates how the four key optimization techniques would work
    together during autoregressive decoding in a transformer-based LLM:
    
    1. Edge-Cloud Collaborative Inference
    2. Runtime Token Pruning
    3. Layer Compression and Skipping
    4. Model-Aware Memory Management
    
    Args:
        input_tokens: The input token IDs
        max_new_tokens: Maximum number of tokens to generate
        model_config: Optional model configuration
        enable_token_pruning: Whether to enable token pruning
        enable_layer_optimization: Whether to enable layer compression/skipping
        enable_memory_management: Whether to enable memory optimizations
        enable_edge_cloud: Whether to enable edge-cloud partitioning
        detailed_logging: Whether to produce detailed logs
        simulated_base_latency_ms: Fake base latency for a token through all layers
        
    Returns:
        A dictionary with inference statistics and simulated results
    """
    # Start timing
    start_time = time.time()
    step_times = []
    
    # Set up default model config if not provided
    if model_config is None:
        model_config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "batch_size": 1
        }
    
    # Extract model dimensions
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_hidden_layers"]
    num_heads = model_config["num_attention_heads"]
    batch_size = model_config["batch_size"]
    
    logger.info("=====================================================")
    logger.info("HYBRID INFERENCE DEMONSTRATION - MULTI-PRONGED APPROACH")
    logger.info("=====================================================")
    logger.info(f"Model dimensions: {hidden_size}d, {num_layers} layers, {num_heads} attention heads")
    logger.info(f"Enabled optimizations:")
    logger.info(f"  - Token Pruning: {'✅' if enable_token_pruning else '❌'}")
    logger.info(f"  - Layer Compression/Skipping: {'✅' if enable_layer_optimization else '❌'}")
    logger.info(f"  - Memory Management: {'✅' if enable_memory_management else '❌'}")
    logger.info(f"  - Edge-Cloud Partitioning: {'✅' if enable_edge_cloud else '❌'}")
    logger.info("=====================================================")
    
    # Initialize the four components
    logger.info("Initializing optimization components...")
    
    # Memory Manager for KV cache
    memory_manager = MemoryManager(
        max_memory_mb=100,  # Simulated max memory
        page_size=8,        # Tokens per page
        quantization_enabled=enable_memory_management,
        memory_threshold_percent=90,
        offloading_enabled=enable_memory_management,
        enable_logging=detailed_logging
    )
    
    # Edge-Cloud Manager for layer partitioning
    edge_cloud_manager = EdgeCloudManager(
        model=None,  # We're not using an actual model here
        device_monitor=None,  # Would be a real device monitor in practice
        cloud_client=None,    # Would be a real cloud client in practice
        mini_llm=None,        # Would be a real mini-LLM handler in practice
        privacy_protection=None,  # Would be real privacy protection in practice
        energy_weight=0.3,
        latency_weight=0.4,
        memory_weight=0.3
    )
    
    # Token Pruner
    token_pruner = TokenPruner(
        pruning_threshold=0.3,  # Less aggressive pruning threshold
        max_shadow_size=100      # Maximum number of tokens to keep in shadow set
    )
    
    # Layer Compression and Skipping
    layer_handler = LayerCompressionAndSkipping(
        model=None,  # In a real implementation, this would be the actual model
        compression_rank=8,
        hot_path_indices=[0, 1, num_layers-1],  # First, second, and last layers are critical
        skip_threshold=0.3
    )
    
    # Prepare for autoregressive generation
    current_tokens = input_tokens.clone()
    seq_len = current_tokens.shape[1]
    
    # Create statistics to track
    stats = {
        "tokens_pruned": 0,
        "tokens_reintroduced": 0,
        "layers_skipped": 0,
        "layers_compressed": 0,
        "cloud_offloaded_layers": 0,
        "local_processed_layers": 0,
        "memory_saved_mb": 0,
        "memory_before_optimization_mb": 0,
        "memory_after_optimization_mb": 0,
        "total_tokens_processed": seq_len,
        "estimated_latency_savings_ms": 0.0,
        "estimated_energy_savings_percent": 0.0,
        "step_details": []
    }
    
    # Pre-compress some layers when initializing
    # In a real implementation, this would actually compress the model
    # For simulation, we'll just track which layers would be compressed
    if enable_layer_optimization:
        # Create a record of compressed layer weights for simulation
        for idx in range(num_layers):
            if idx not in layer_handler.hot_path_indices or idx == layer_handler.hot_path_indices[-1]:
                # Track this layer as compressed in our simulator
                layer_handler.compressed_layers[idx] = {
                    'original_shape': (hidden_size, hidden_size * 4),  # typical MLP expansion
                    'compressed': True,
                    'rank': layer_handler.compression_rank,
                    'compression_ratio': layer_handler.compression_rank * (hidden_size + hidden_size * 4) / (hidden_size * hidden_size * 4)
                }
                stats["layers_compressed"] += 1
                
        # Initialize layer temperatures to simulate different layer importances
        for layer_idx in range(num_layers):
            # Hot path layers get high temperature
            if layer_idx in layer_handler.hot_path_indices:
                temp = 0.9
            # Earlier and later layers tend to be more important than middle ones
            elif layer_idx < num_layers * 0.3:
                temp = 0.7 - 0.3 * (layer_idx / (num_layers * 0.3))
            elif layer_idx > num_layers * 0.7:
                temp = 0.4 + 0.3 * ((layer_idx - num_layers * 0.7) / (num_layers * 0.3))
            else:
                # Middle layers get lower temperatures - more likely to be skipped
                temp = 0.2 + 0.1 * random.random()
                
            layer_handler.layer_temperatures[layer_idx] = temp
    
    logger.info(f"Starting inference with input sequence length: {seq_len}")
    
    # Main autoregressive generation loop
    for i in range(max_new_tokens):
        step_start_time = time.time()
        logger.info(f"\n--- Generation step {i+1}/{max_new_tokens} ---")
        
        # Statistics for this step
        step_stats = {
            "step": i+1,
            "seq_len": current_tokens.shape[1],
            "tokens_pruned": 0,
            "layers_skipped": 0,
            "layers_compressed": 0,
            "cloud_offloaded": 0,
            "memory_usage_mb": 0,
            "estimated_latency_ms": simulated_base_latency_ms
        }
        
        # 1. Memory Management: Update KV cache
        current_seq_len = current_tokens.shape[1]
        key_value_tensors = {
            "key": create_fake_hidden_state(batch_size, current_seq_len, hidden_size),
            "value": create_fake_hidden_state(batch_size, current_seq_len, hidden_size)
        }
        
        # Track memory usage before optimizations
        pre_mem_usage = memory_manager.calculate_memory_usage()
        stats["memory_before_optimization_mb"] += pre_mem_usage
        
        # Add the token positions to the KV cache
        token_indices = list(range(current_seq_len))
        # Process each layer (in a real implementation, we'd have per-layer KV states)
        for layer_idx in range(num_layers):
            memory_manager.add_to_kv_cache(
                layer_idx,
                key_value_tensors["key"],
                key_value_tensors["value"],
                token_indices
            )
        
        # Force compression of pages every other step
        if i % 2 == 0 and enable_memory_management:
            logger.info("Forcing memory compression for demonstration...")
            # Get candidate pages for compression
            candidates = []
            for page_key in memory_manager.onchip_pages:
                if not memory_manager.page_metadata[page_key].get('compressed', False):
                    candidates.append(page_key)
            
            # Compress up to 5 pages
            compressed_count = 0
            total_memory_saved = 0
            for layer_idx, page_idx in candidates[:5]:
                memory_saved = memory_manager.compress_page(layer_idx, page_idx)
                if memory_saved > 0:
                    compressed_count += 1
                    total_memory_saved += memory_saved
            
            if compressed_count > 0:
                logger.info(f"Compressed {compressed_count} pages, saved {total_memory_saved/1024:.2f}KB")

        mem_usage = memory_manager.calculate_memory_usage()
        step_stats["memory_usage_mb"] = mem_usage
        stats["memory_after_optimization_mb"] += mem_usage
        
        # Calculate memory saved in this step
        memory_saved = max(0, pre_mem_usage - mem_usage)
        stats["memory_saved_mb"] += memory_saved
        
        total_pages = len(memory_manager.onchip_pages) + len(memory_manager.offchip_pages)
        logger.info(f"Memory usage: {mem_usage:.2f}MB, Total pages: {total_pages}")
        
        # Get detailed compression stats
        if enable_memory_management:
            compression_stats = memory_manager.get_compression_stats()
            if compression_stats["compressed_pages"] > 0:
                logger.info(f"Memory compression: {compression_stats['original_size_mb']:.2f}MB → "
                           f"{compression_stats['compressed_size_mb']:.2f}MB "
                           f"({compression_stats['percent_saved']:.1f}% saved)")
                step_stats["compression_ratio"] = compression_stats["compression_ratio"]
                step_stats["memory_saved_from_compression_mb"] = compression_stats["memory_saved_mb"]
                # Update overall memory savings
                stats["memory_saved_mb"] += compression_stats["memory_saved_mb"] / max_new_tokens
        
        # 1. Token Pruning
        pruned_indices = []
        if enable_token_pruning:
            # Create attention matrix and hidden states for token scoring
            attention_matrix = create_fake_attention_matrix(current_seq_len)
            hidden_states = create_fake_hidden_states(batch_size, current_seq_len, hidden_size)
            
            token_scores = token_pruner.score_tokens(attention_matrix, list(range(current_seq_len))) 
            pruned_indices = token_pruner.identify_prunable_tokens()
            
            if pruned_indices:
                # Create boolean mask for tokens to keep
                mask = torch.ones(current_seq_len, dtype=torch.bool)
                mask[pruned_indices] = False
                
                # Apply mask to attention matrix (both dimensions)
                attention_matrix = attention_matrix[:, :, ~mask][:, :, :, ~mask]
                
                # Update hidden states
                if isinstance(hidden_states, torch.Tensor):
                    hidden_states = hidden_states[:, ~mask]
                elif isinstance(hidden_states, tuple):
                    hidden_states = tuple(h[:, ~mask] for h in hidden_states)
                
                # Update sequence length
                current_seq_len = int((~mask).sum())
                
                # Log pruning results
                print(f"Pruned {len(pruned_indices)} tokens, new sequence length: {current_seq_len}")
                
                # Update token indices for memory manager
                token_indices = np.array(token_indices)
                token_indices = token_indices[~mask.cpu().numpy()]
                token_indices = token_indices.tolist()
                
                # Update statistics
                stats["tokens_pruned"] += len(pruned_indices)
                step_stats["tokens_pruned"] = len(pruned_indices)
            else:
                logger.info("No tokens pruned in this step")
        
        # 3. Edge-Cloud Layer Partitioning: Simulate network and device conditions
        network_conditions = simulate_network_conditions()
        device_load = simulate_device_load()
        
        # Determine layer partitioning (which layers run where)
        local_layers, remote_layers = [], []
        if enable_edge_cloud:
            # For demonstration purposes, we'll force some layers to run locally
            # so we can test layer skipping
            
            # Normally this would come from edge_cloud_manager
            # local_layers, remote_layers = edge_cloud_manager.determine_full_partition()
            
            # Force certain layers to run locally based on step number
            # This simulates changing network conditions
            if i % 2 == 0:  # On even steps, run more layers locally
                # Run 1/3 of layers locally
                local_layers = list(range(0, num_layers, 3))
                remote_layers = [i for i in range(num_layers) if i not in local_layers]
            else:  # On odd steps, run fewer layers locally
                # Run every 4th layer locally
                local_layers = list(range(0, num_layers, 4))
                remote_layers = [i for i in range(num_layers) if i not in local_layers]
            
            # Log the decision
            logger.info(f"Decided partition: {len(local_layers)} layers local, {len(remote_layers)} layers remote")
        else:
            # If edge-cloud is disabled, run everything locally
            local_layers = list(range(num_layers))
        
        # Update statistics
        stats["local_processed_layers"] += len(local_layers)
        stats["cloud_offloaded_layers"] += len(remote_layers)
        step_stats["cloud_offloaded"] = len(remote_layers)
        
        # Log partitioning decision
        logger.info(f"Network bandwidth: {network_conditions['bandwidth_mbps']:.2f} Mbps")
        logger.info(f"Processing {len(local_layers)} layers locally, {len(remote_layers)} layers in cloud")
        
        # 4. Layer Compression and Skipping: Decide which layers to compress or skip
        compressed_layers = []
        skipped_layers = []
        layer_decisions = {}
        
        if enable_layer_optimization:
            # Process each layer based on if it's local or remote
            for layer_idx in range(num_layers):
                if layer_idx in remote_layers:
                    # For cloud layers, we don't need to compress/skip
                    layer_decisions[layer_idx] = "cloud"
                else:
                    # For local layers, decide compression/skipping strategy
                    # Create a fake hidden state for the gating function
                    fake_hidden_state = create_fake_hidden_state(batch_size, current_seq_len, hidden_size)
                    
                    # Check if we should skip this layer
                    should_skip = layer_handler.should_skip_layer(layer_idx, fake_hidden_state)
                    
                    if should_skip:
                        skipped_layers.append(layer_idx)
                        stats["layers_skipped"] += 1
                        step_stats["layers_skipped"] += 1
                        layer_decisions[layer_idx] = "skip"
                    else:
                        # If not skipping, see if we should compress
                        is_compressed = layer_idx in layer_handler.compressed_layers
                        if is_compressed:
                            compression_rate = layer_handler.compressed_layers[layer_idx].get('compression_ratio', 0.25)
                            compressed_layers.append((layer_idx, compression_rate))
                            step_stats["layers_compressed"] += 1
                            layer_decisions[layer_idx] = f"compress_{compression_rate:.2f}"
                        else:
                            layer_decisions[layer_idx] = "full"
        else:
            # If layer optimization is disabled, process all local layers fully
            for layer_idx in range(num_layers):
                if layer_idx in remote_layers:
                    layer_decisions[layer_idx] = "cloud"
                else:
                    layer_decisions[layer_idx] = "full"
        
        # Log layer handling decisions
        if skipped_layers:
            logger.info(f"Skipped layers: {skipped_layers}")
        if compressed_layers:
            logger.info(f"Compressed layers: {[idx for idx, _ in compressed_layers]}")
            
        # 5. Simulate processing through the network
        # In a real implementation, this would be where we execute the actual model layers
        if detailed_logging:
            logger.info("Processing layers with the following strategies:")
            for layer_idx in range(num_layers):
                strategy = layer_decisions.get(layer_idx, "unknown")
                logger.info(f"  Layer {layer_idx}: {strategy}")
        
        # 6. Simulate generating a new token
        # In a real implementation, this would come from the model output
        new_token = torch.tensor([[random.randint(0, 50000)]])
        current_tokens = torch.cat([current_tokens, new_token], dim=1)
        
        # Update the total token count
        stats["total_tokens_processed"] += 1
        
        # 7. Periodically reintroduce pruned tokens (token reuse)
        if enable_token_pruning and i % 3 == 0 and token_pruner.shadow_set:
            # In a real implementation we'd pass current tokens and states
            # Here we'll just pass the number of tokens we want to reintroduce
            reintroduced = token_pruner.reintroduce_tokens(current_tokens.tolist(), None)
            if reintroduced:
                stats["tokens_reintroduced"] += len(reintroduced)
                logger.info(f"Reintroduced {len(reintroduced)} tokens from the shadow set")
        
        # 8. Estimate performance benefits
        latency_savings = estimate_latency_savings(
            simulated_base_latency_ms,
            step_stats["tokens_pruned"],
            step_stats["layers_skipped"],
            step_stats["layers_compressed"],
            current_seq_len,
            num_layers
        )
        
        stats["estimated_latency_savings_ms"] += latency_savings
        step_stats["estimated_latency_ms"] = simulated_base_latency_ms - latency_savings
        
        # Calculate approximate energy savings (very rough estimate)
        energy_savings_percent = 0.0
        if current_seq_len > 0:
            token_pruning_factor = min(0.5, step_stats["tokens_pruned"] / current_seq_len) * 0.4
        else:
            token_pruning_factor = 0.0
            
        layer_skip_factor = (step_stats["layers_skipped"] / max(1, num_layers)) * 0.3
        layer_compress_factor = (step_stats["layers_compressed"] / max(1, num_layers)) * 0.2
        cloud_factor = (step_stats["cloud_offloaded"] / max(1, num_layers)) * 0.5
        
        energy_savings_percent = (token_pruning_factor + layer_skip_factor + 
                                 layer_compress_factor + cloud_factor) * 100
        
        step_stats["energy_savings_percent"] = energy_savings_percent
        stats["estimated_energy_savings_percent"] += energy_savings_percent / max_new_tokens
        
        # Record step completion time
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_times.append(step_duration)
        step_stats["actual_processing_time"] = step_duration
        
        # Add step stats to the detailed log
        stats["step_details"].append(step_stats)
        
        # Real-time efficiency report
        logger.info(f"Step efficiency: {energy_savings_percent:.1f}% energy saving, " +
                  f"{latency_savings:.1f}ms latency saving")
    
    # Calculate final statistics
    end_time = time.time()
    total_duration = end_time - start_time
    avg_token_time = sum(step_times) / len(step_times) if step_times else 0
    
    # Summarize the inference process
    logger.info("\n=== Inference Summary ===")
    logger.info(f"Total processing time: {total_duration:.2f}s")
    logger.info(f"Average time per token: {avg_token_time:.4f}s")
    logger.info(f"Generated {max_new_tokens} new tokens")
    logger.info(f"Final sequence length: {current_tokens.shape[1]}")
    logger.info("\nOptimization Statistics:")
    logger.info(f"• Tokens pruned: {stats['tokens_pruned']} ({stats['tokens_pruned']/stats['total_tokens_processed']*100:.1f}% of total)")
    logger.info(f"• Tokens reintroduced: {stats['tokens_reintroduced']}")
    logger.info(f"• Layers skipped: {stats['layers_skipped']}")
    logger.info(f"• Layers compressed: {stats['layers_compressed']}")
    logger.info(f"• Layers processed locally: {stats['local_processed_layers']}")
    logger.info(f"• Layers offloaded to cloud: {stats['cloud_offloaded_layers']}")

    # Memory optimization summary
    if enable_memory_management:
        # Get final compression stats
        final_compression_stats = memory_manager.get_compression_stats()
        avg_mem_before = stats["memory_before_optimization_mb"] / max_new_tokens if max_new_tokens > 0 else 0
        avg_mem_after = stats["memory_after_optimization_mb"] / max_new_tokens if max_new_tokens > 0 else 0
        logger.info(f"• Memory usage: {avg_mem_before:.2f}MB → {avg_mem_after:.2f}MB (avg per step)")
        logger.info(f"• Memory savings from compression: {final_compression_stats['memory_saved_mb']:.2f}MB " +
                   f"({final_compression_stats['percent_saved']:.1f}% compression ratio)")
        logger.info(f"• Total memory saved: {stats['memory_saved_mb']:.2f}MB")
    else:
        logger.info(f"• Estimated memory saved: {stats['memory_saved_mb']:.2f} MB")

    logger.info(f"• Estimated latency savings: {stats['estimated_latency_savings_ms']:.2f} ms")
    logger.info(f"• Estimated energy savings: {stats['estimated_energy_savings_percent']:.1f}%")
    
    return {
        "final_tokens": current_tokens,
        "stats": stats,
        "total_time": total_duration,
        "avg_token_time": avg_token_time
    }

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run hybrid inference demonstration")
    parser.add_argument("--tokens", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--seq-len", type=int, default=5, help="Initial sequence length")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    
    # Optimization flags
    parser.add_argument("--no-token-pruning", action="store_true", help="Disable token pruning")
    parser.add_argument("--no-layer-opt", action="store_true", help="Disable layer optimization")
    parser.add_argument("--no-memory-opt", action="store_true", help="Disable memory optimization")
    parser.add_argument("--no-edge-cloud", action="store_true", help="Disable edge-cloud partitioning")
    
    # Output options
    parser.add_argument("--detailed", action="store_true", help="Enable detailed logging")
    parser.add_argument("--save-stats", action="store_true", help="Save statistics to JSON file")
    parser.add_argument("--output", type=str, default="", help="Output file for statistics")
    
    args = parser.parse_args()
    
    # Create a simple test case with the specified sequence length
    input_ids = torch.tensor([[101] + [2000 + i for i in range(args.seq_len - 1)]])
    
    # Configure model dimensions
    model_config = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "batch_size": 1
    }
    
    # Run the simulated inference
    logger.info("Starting hybrid inference demonstration...")
    result = run_inference(
        input_ids, 
        max_new_tokens=args.tokens,
        model_config=model_config,
        enable_token_pruning=not args.no_token_pruning,
        enable_layer_optimization=not args.no_layer_opt,
        enable_memory_management=not args.no_memory_opt,
        enable_edge_cloud=not args.no_edge_cloud,
        detailed_logging=args.detailed
    )
    
    # Display final statistics
    logger.info("\nFinal Statistics Summary:")
    for key, value in result["stats"].items():
        if key != "step_details":  # Skip the detailed step info
            logger.info(f"{key}: {value}")
            
    # Save statistics if requested
    if args.save_stats:
        output_file = args.output if args.output else f"hybrid_inference_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result["stats"], f, indent=2)
        logger.info(f"Statistics saved to {output_file}") 