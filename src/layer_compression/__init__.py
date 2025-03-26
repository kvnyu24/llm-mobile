"""
Layer Compression and Skipping module for on-device LLM inference.

This module implements the third optimization technique described in the research paper:
- Layerwise Compression and Skipping

It provides classes for reducing computational cost by applying low-rank factorization
to model weights and dynamically skipping layers during inference.
"""

from .layer_compression_skipping import LayerCompressionAndSkipping

__all__ = ["LayerCompressionAndSkipping"] 