"""
Memory Manager module for on-device LLM inference.

This module implements the fourth optimization technique described in the research paper:
- Model-Aware Memory Management

It provides classes for efficient memory usage through paged KV cache management,
compression, eviction, and integration with token pruning.
"""

from .memory_manager import MemoryManager

__all__ = ["MemoryManager"] 