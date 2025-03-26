# Runtime Token Pruning

This module implements runtime token pruning for efficient LLM inference on mobile devices, as described in the paper.

## Overview

Token pruning improves inference efficiency by identifying and removing "low-impact" tokens during autoregressive decoding. This technique:

- Reduces the quadratic cost of attention computation by physically removing tokens that contribute little to the output
- Maintains a "shadow set" of pruned tokens that can be reintroduced if they become relevant later
- Provides significant performance benefits for long context scenarios

## Implementation Details

The implementation includes:

- `TokenPruner` class for managing token pruning
- Attention-based scoring to identify unimportant tokens
- Configurable pruning threshold
- Shadow set mechanism to store pruned tokens
- Metrics tracking for pruning statistics

## Usage

Here's a simple example of how to use the token pruner:

```python
from token_pruning import TokenPruner
import torch

# Create a token pruner with desired threshold
pruner = TokenPruner(pruning_threshold=0.01)

# During generation, score tokens using attention weights
attention_weights = model_output.attentions[-1]  # Last layer's attention
token_indices = list(range(len(current_tokens)))
token_scores = pruner.score_tokens(attention_weights, token_indices)

# Prune low-importance tokens
pruned_tokens, pruned_hidden_states = pruner.prune_tokens(
    current_tokens, hidden_states
)

# Check pruning statistics
stats = pruner.get_pruning_stats()
print(f"Pruned {stats['total_tokens_pruned']} tokens out of {stats['total_tokens_seen']}")
```

## Integration with Transformers

To fully integrate token pruning with transformer models, you need to:

1. Extract attention weights from model outputs
2. Score tokens based on these attention weights
3. Prune tokens and update hidden states
4. Update the model's key-value cache to reflect pruned tokens (model-specific)

A complete example is provided in `src/examples/token_pruning_example.py`.

## Performance Impact

The performance impact depends on the sequence length and pruning threshold:

- For short sequences (<50 tokens), the overhead may outweigh benefits
- For long sequences (>100 tokens), pruning can significantly reduce computation
- Typical pruning ratios range from 20% to 40% of tokens
- The speedup increases with sequence length due to the quadratic nature of attention

## References

This implementation is based on the techniques described in the paper on efficient on-device LLM inference. 