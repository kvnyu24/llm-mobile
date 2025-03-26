# Layer Compression and Skipping

This module implements the Layerwise Compression and Skipping technique for efficient on-device LLM inference as described in the research paper.

## Overview

The Layer Compression and Skipping technique combines two strategies to reduce the computational cost of running Transformer models:

1. **Low-Rank Factorization (Compression)**: Weight matrices are decomposed into a combination of a smaller base matrix and low-rank factors: `W = W₀ + BA`, where B and A are rank-r matrices. This drastically reduces the parameter count while preserving most of the model's capabilities.

2. **Dynamic Layer Skipping**: A gating function determines which layers can be skipped during inference based on the current input and computational constraints. Layers are divided into:
   - **Hot Path**: Critical layers that are never skipped (typically early and last layers)
   - **Cold Path**: Layers that can be optionally skipped based on their "temperature" (importance)

## Key Features

- Configurable low-rank factorization with adjustable rank
- Temperature-based gating for dynamic layer skipping
- Hot path preservation for critical layers
- Adaptive compression and skipping based on available compute
- Performance metrics tracking
- Compatible with various transformer architectures

## Usage

```python
from layer_compression.layer_compression_skipping import LayerCompressionAndSkipping

# Initialize with a transformer model
lcs = LayerCompressionAndSkipping(
    model=model,
    compression_rank=8,  # Lower = more compression but less accuracy
    hot_path_indices=[0, 1, model.config.num_hidden_layers-1],  # Never skip these
    skip_threshold=0.3  # Skip layers with temperature below this
)

# Apply compression
lcs.apply_low_rank_factorization()

# Add layer skipping capability
lcs.update_model_with_gating()

# Adapt to available compute (0.0 = minimal, 1.0 = abundant)
lcs.adjust_compression_level(available_compute=0.5)

# Run inference (skipping happens automatically)
outputs = model(input_ids)

# Get performance metrics
metrics = lcs.get_metrics()
```

## Example

See `src/examples/layer_compression_example.py` for a complete example of using this technique with transformer models.

You can run the example using the provided script:

```bash
# Run with default parameters
./src/run_layer_compression.sh

# Run with custom parameters
./src/run_layer_compression.sh --model gpt2 --rank 4 --threshold 0.4 --compute 0.3
```

## Implementation Details

### Layer Temperature

Layer "temperature" is a measure of how important a layer is for the current input. Higher temperature indicates that a layer is more critical for accurate prediction. The temperature calculation can be based on:

- Activation patterns in the hidden states
- Gradient-based importance
- Heuristics from prior knowledge of the model

### Low-Rank Factorization

The implementation uses a LoRA-style approach where large weight matrices are decomposed into:
- Original weights W₀ (kept for layers requiring high accuracy)
- Low-rank factors B and A for compressed layers

### Gating Function

The gating function `g(·)` determines whether to skip a layer based on:
- Whether the layer is in the hot path (never skipped)
- The layer's temperature compared to the skip threshold
- The current computational constraints

## Adjusting for Performance

- **Lower rank**: More compression, less accuracy
- **Higher skip threshold**: More layers skipped, faster inference, potential accuracy loss
- **Hot path size**: More hot path layers = higher accuracy but less optimization 