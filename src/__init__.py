"""
LLM Mobile - Efficient LLM Inference on Mobile Devices

This package implements four key optimization techniques for running LLM inference 
on resource-constrained mobile devices:

1. Edge-Cloud Collaborative Inference
2. Runtime Token Pruning
3. Layerwise Compression and Skipping
4. Model-Aware Memory Management
"""

from .hybrid_inference import run_inference, HybridInferenceEngine

__all__ = ["run_inference", "HybridInferenceEngine"] 