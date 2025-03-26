# LLM Mobile: Efficient LLM Inference on Mobile Devices

This project implements the framework described in a research paper for running Large Language Model (LLM) inference efficiently on mobile devices with limited resources.

## Overview

The architecture consists of four key optimization techniques:

1. **Edge-Cloud Collaborative Inference**
   - Dynamically splits model layers between device and cloud
   - Uses a small local model for "easy" tokens and remote model for "hard" tokens
   - Includes encryption and dimensionality reduction for privacy

2. **Runtime Token Pruning**
   - Removes low-impact tokens during inference to reduce sequence length
   - Maintains a "shadow set" of removed tokens that can be reintroduced
   - Reduces the quadratic cost of attention as sequences grow

3. **Layerwise Compression and Skipping**
   - Applies low-rank factorization (LoRA-style) to reduce parameter size
   - Dynamically skips "cold" layers using a gating function
   - Maintains a "hot path" of essential layers that are never skipped

4. **Model-Aware Memory Management**
   - Organizes KV cache into "pages" for efficient management
   - Applies eviction strategies and compression for older pages
   - Integrates with token pruning and edge-cloud offloading

## Project Structure

```
src/
├── edge_cloud/
│   └── edge_cloud_manager.py         # Handles device-cloud split
├── token_pruning/
│   └── token_pruner.py               # Manages token pruning
├── layer_compression/
│   └── layer_compression_skipping.py # Handles layer optimization
├── memory_manager/
│   └── memory_manager.py             # Manages memory usage
└── hybrid_inference.py               # Coordinates all optimizations
```

## Usage

Basic usage:

```python
from hybrid_inference import run_inference
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")  # Example model

result = run_inference(
    model=model,
    input_text="Hello, world!",
    edge_cloud_config={
        "bandwidth_threshold": 1.0,  # Mbps
    },
    token_pruning_config={
        "pruning_threshold": 0.01,
        "max_shadow_size": 100,
    },
    layer_compression_config={
        "compression_rank": 8,
        "hot_path_indices": [0, 1, 11],  # First, second, and last layers
    },
    memory_manager_config={
        "max_memory_mb": 512,
        "page_size": 16,
        "quantization_enabled": True,
    },
    max_new_tokens=100
)
```

## Requirements

- Python 3.8+
- PyTorch
- transformers

## Future Work

This is currently a skeleton implementation. Future work includes:
- Implementing the actual core functionality of each component
- Adding benchmarks to measure memory and compute savings
- Integration with popular mobile frameworks 