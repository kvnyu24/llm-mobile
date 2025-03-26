#!/usr/bin/env python
"""
Memory Manager Example

This example demonstrates how to use the MemoryManager class to efficiently
manage memory for LLM inference on resource-constrained devices through
techniques like paged KV cache, compression, and eviction.
"""

import os
import sys
import time
import numpy as np
import argparse

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory_manager.memory_manager import MemoryManager

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    print("PyTorch or transformers not available. Using mock implementations.")
    HAS_TORCH = False

class MockModelConfig:
    """Mock model configuration for demonstration when no model is available."""
    def __init__(self):
        self.num_hidden_layers = 12
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.head_dim = self.hidden_size // self.num_attention_heads

def generate_random_tensors(batch_size, seq_len, hidden_dim, num_heads, head_dim):
    """Generate random tensors for key and value in KV cache."""
    if HAS_TORCH:
        # PyTorch tensors [batch_size, seq_len, num_heads, head_dim]
        keys = torch.randn(batch_size, seq_len, num_heads, head_dim)
        values = torch.randn(batch_size, seq_len, num_heads, head_dim)
    else:
        # NumPy arrays
        keys = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
        values = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
    
    return keys, values

def run_example(args):
    print("\n=== Memory Manager Example ===\n")
    
    # Create memory manager
    memory_manager = MemoryManager(
        max_memory_mb=args.max_memory,
        page_size=args.page_size,
        quantization_enabled=args.quantization,
        memory_threshold_percent=args.threshold,
        offloading_enabled=args.offloading
    )
    
    # Initialize with model configuration
    if HAS_TORCH and args.use_model:
        try:
            model_name = args.model_name
            print(f"Loading model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model_config = model.config
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to mock model configuration")
            model_config = MockModelConfig()
    else:
        print("Using mock model configuration")
        model_config = MockModelConfig()
    
    # Initialize KV cache
    memory_manager.initialize_kv_cache(model_config)
    
    # Setup dimensions for demo
    batch_size = 1
    sequence_length = 100  # We'll generate a sequence of this length
    num_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // num_heads
    
    print(f"\nSimulating inference with sequence length {sequence_length}")
    print(f"Model dimensions: layers={model_config.num_hidden_layers}, " +
          f"heads={num_heads}, head_dim={head_dim}")
    
    # Simulate autoregressive generation
    print("\nPhase 1: Initial sequence processing (adding tokens to KV cache)")
    for step in range(0, min(sequence_length, 50), 5):  # Process in chunks for demo purposes
        # Simulate processing a chunk of the sequence
        chunk_size = 5
        seq_chunk = list(range(step, step + chunk_size))
        
        print(f"  Processing tokens {seq_chunk}")
        
        # For each layer, add key/value tensors to cache
        for layer_idx in range(model_config.num_hidden_layers):
            # Generate random key/value tensors for this chunk
            keys, values = generate_random_tensors(
                batch_size=batch_size,
                seq_len=chunk_size,
                hidden_dim=model_config.hidden_size,
                num_heads=num_heads,
                head_dim=head_dim
            )
            
            # Add to KV cache
            memory_manager.add_to_kv_cache(
                layer_idx=layer_idx,
                key=keys,
                value=values,
                token_indices=seq_chunk
            )
        
        # Print memory usage periodically
        if step % 10 == 0 or step == 45:
            current_usage = memory_manager.calculate_memory_usage()
            print(f"  Memory usage after token {step+chunk_size-1}: {current_usage:.2f} MB")
    
    # Print memory statistics after initial sequence
    stats = memory_manager.get_stats()
    print("\nMemory statistics after initial sequence:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Memory usage: {stats['current_memory_usage_mb']:.2f} MB")
    print(f"  Compressed pages: {stats['compressed_pages']}")
    
    # Phase a simulated token pruning operation
    print("\nPhase 2: Simulating token pruning (removing less important tokens)")
    
    # Identify "less important" tokens to prune (for demo, we'll use every 3rd token)
    tokens_to_prune = list(range(2, 50, 3))
    print(f"  Pruning {len(tokens_to_prune)} tokens: {tokens_to_prune[:10]}...")
    
    # Prune tokens
    bytes_saved = memory_manager.remove_pruned_tokens(tokens_to_prune)
    print(f"  Pruned {len(tokens_to_prune)} tokens, saved {bytes_saved/1024:.2f} KB")
    
    # Print memory usage after pruning
    current_usage = memory_manager.calculate_memory_usage()
    print(f"  Memory usage after pruning: {current_usage:.2f} MB")
    
    # Phase 3: Force manual compression
    print("\nPhase 3: Demonstrating manual compression")
    
    # Compress a few specific pages
    compressed_pages = 0
    for layer_idx in range(0, model_config.num_hidden_layers, 2):  # Every other layer
        for page_idx in range(0, 3):  # First few pages
            bytes_saved = memory_manager.compress_page(layer_idx, page_idx)
            if bytes_saved > 0:
                compressed_pages += 1
    
    print(f"  Manually compressed {compressed_pages} pages")
    current_usage = memory_manager.calculate_memory_usage()
    print(f"  Memory usage after manual compression: {current_usage:.2f} MB")
    
    # Phase 4: Continue sequence to trigger automatic memory management
    print("\nPhase 4: Continuing sequence to trigger automatic memory management")
    
    # Force a smaller memory limit to trigger eviction
    if args.force_eviction:
        print("  Forcing lower memory limit to trigger eviction")
        old_limit = memory_manager.max_memory_mb
        memory_manager.max_memory_mb = current_usage * 0.9  # 90% of current usage
        print(f"  New memory limit: {memory_manager.max_memory_mb:.2f} MB (was {old_limit} MB)")
    
    # Continue generating the sequence
    for step in range(50, sequence_length, 1):  # One token at a time now
        # For each layer, add key/value tensors to cache
        for layer_idx in range(model_config.num_hidden_layers):
            # Generate random key/value tensors for this token
            keys, values = generate_random_tensors(
                batch_size=batch_size,
                seq_len=1,
                hidden_dim=model_config.hidden_size,
                num_heads=num_heads,
                head_dim=head_dim
            )
            
            # Add to KV cache
            memory_manager.add_to_kv_cache(
                layer_idx=layer_idx,
                key=keys,
                value=values,
                token_indices=[step]
            )
        
        # Print memory usage periodically
        if step % 10 == 0:
            current_usage = memory_manager.calculate_memory_usage()
            print(f"  Memory usage after token {step}: {current_usage:.2f} MB")
    
    # Print final memory statistics
    stats = memory_manager.get_stats()
    print("\nFinal memory statistics:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Compressed pages: {stats['compressed_pages']}")
    print(f"  Evicted pages: {stats['evicted_pages']}")
    print(f"  Offloaded pages: {stats['offloaded_pages']}")
    print(f"  Pruned tokens: {stats['pruned_tokens']}")
    print(f"  Memory savings: {stats['memory_savings_mb']:.2f} MB")
    print(f"  Current memory usage: {stats['current_memory_usage_mb']:.2f} MB")
    print(f"  On-chip pages: {stats['onchip_pages']}")
    print(f"  Off-chip pages: {stats['offchip_pages']}")
    
    # Phase 5: Retrieve tokens to demonstrate fetching from cache
    print("\nPhase 5: Retrieving tokens from cache")
    
    # Try retrieving a mix of tokens (some may have been evicted)
    token_indices_to_retrieve = [10, 20, 30, 40, 60, 70, 80, 90]
    print(f"  Retrieving tokens: {token_indices_to_retrieve}")
    
    for layer_idx in range(0, model_config.num_hidden_layers, 3):  # Sample a few layers
        keys, values = memory_manager.get_from_kv_cache(layer_idx, token_indices_to_retrieve)
        if keys is not None:
            if HAS_TORCH:
                print(f"  Layer {layer_idx}: Retrieved {keys.shape[1]} tokens")
            else:
                print(f"  Layer {layer_idx}: Retrieved {keys.shape[1]} tokens")
        else:
            print(f"  Layer {layer_idx}: No tokens retrieved")
    
    print("\nExample completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory Manager Example')
    parser.add_argument('--max-memory', type=float, default=50.0, 
                        help='Maximum memory budget in MB (default: 50.0)')
    parser.add_argument('--page-size', type=int, default=16,
                        help='Number of tokens per page (default: 16)')
    parser.add_argument('--threshold', type=float, default=90.0,
                        help='Memory threshold percentage (default: 90.0)')
    parser.add_argument('--quantization', action='store_true', default=True,
                        help='Enable quantization for compression')
    parser.add_argument('--offloading', action='store_true', default=True,
                        help='Enable offloading pages to cloud/disk')
    parser.add_argument('--use-model', action='store_true', default=False,
                        help='Use actual HuggingFace model if available')
    parser.add_argument('--model-name', type=str, default='gpt2',
                        help='HuggingFace model name (default: gpt2)')
    parser.add_argument('--force-eviction', action='store_true', default=True,
                        help='Force eviction by reducing memory limit')
    
    args = parser.parse_args()
    run_example(args) 