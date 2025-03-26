# Model-Aware Memory Manager

This module implements the Model-Aware Memory Management approach for efficient on-device LLM inference as described in the research paper.

## Overview

The Memory Manager efficiently handles memory usage for LLM inference on resource-constrained devices by implementing a paged key-value cache system with dynamic memory management strategies.

## Key Features

1. **Paged KV Cache**: Organizes the attention key-value cache into fixed-size pages of tokens
2. **Dynamic Memory Management**: 
   - Compression (FP16 → INT8 quantization)
   - Least-Recently-Used (LRU) eviction
   - Off-chip offloading
3. **Token Pruning Integration**: Efficiently removes pruned tokens from the cache
4. **Memory Usage Monitoring**: Tracks and manages memory usage to stay within budget

## Core Components

### Paged Memory Organization

The KV cache is organized as follows:
- Each layer has its own set of pages
- Each page contains a fixed number of tokens (configurable)
- Pages track which tokens they contain
- The system knows which pages are on-chip vs. off-chip

### Memory Management Strategies

When memory usage exceeds the configured threshold:

1. **Compression** is tried first:
   - Pages are compressed using quantization (FP16 → INT8)
   - Dequantization happens automatically when retrieving tokens

2. **Eviction/Offloading** is used if compression isn't enough:
   - Least recently used pages are identified
   - Pages can be offloaded to cloud/disk or completely evicted
   - Metadata tracks page locations for potential retrieval

3. **Token Pruning Integration**:
   - Tokens identified by the token pruning system can be removed
   - Pages with all tokens pruned are completely freed

## Usage

```python
from memory_manager.memory_manager import MemoryManager

# Initialize the memory manager
memory_manager = MemoryManager(
    max_memory_mb=512,  # Maximum memory budget
    page_size=16,       # Tokens per page
    quantization_enabled=True,
    memory_threshold_percent=90  # When to trigger management
)

# Initialize with model configuration
memory_manager.initialize_kv_cache(model_config)

# During inference, add to KV cache
memory_manager.add_to_kv_cache(
    layer_idx=layer_idx,
    key=key_tensor,
    value=value_tensor,
    token_indices=[token_idx]
)

# Retrieve from KV cache
keys, values = memory_manager.get_from_kv_cache(
    layer_idx=layer_idx,
    token_indices=[token_idx1, token_idx2, ...]
)

# If token pruning is used
memory_manager.remove_pruned_tokens([token_idx1, token_idx2, ...])

# Get memory usage statistics
stats = memory_manager.get_stats()
```

## Example

An example script is provided in `src/examples/memory_manager_example.py` that demonstrates:
- KV cache initialization
- Adding tokens during autoregressive generation
- Memory management (compression and eviction)
- Token pruning integration
- Retrieval from the cache

You can run the example with:

```bash
./run_memory_manager.sh
```

Use `./run_memory_manager.sh --help` to see available options. 