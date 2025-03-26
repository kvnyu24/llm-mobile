import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging

# Initialize logger
logger = logging.getLogger("memory_manager")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MemoryManager:
    """
    Model-Aware Memory Manager
    
    This class handles efficient memory usage for LLM inference on memory-constrained
    devices through key-value cache management and compression techniques.
    
    Key features:
    - Organizes attention key-value cache into "pages" for efficient management
    - Implements eviction strategies for older pages when memory is constrained
    - Applies quantization to reduce memory footprint (e.g., FP16 → INT8)
    - Integrates with token pruning and edge-cloud offloading for comprehensive memory management
    
    Based on the fourth technique from the research paper on efficient
    on-device LLM inference.
    """
    
    def __init__(self, max_memory_mb=512, page_size=16, quantization_enabled=True, 
                 memory_threshold_percent=90, offloading_enabled=True):
        """
        Initialize the Memory Manager.
        
        Args:
            max_memory_mb: Maximum memory budget in MB
            page_size: Number of tokens per memory page
            quantization_enabled: Whether to enable quantization for memory savings
            memory_threshold_percent: Percentage of max memory that triggers eviction/compression
            offloading_enabled: Whether to allow offloading pages to the cloud
        """
        self.max_memory_mb = max_memory_mb
        self.page_size = page_size
        self.quantization_enabled = quantization_enabled
        self.memory_threshold_percent = memory_threshold_percent
        self.offloading_enabled = offloading_enabled
        
        # KV cache organized by layers and pages
        # Structure: {layer_idx: {page_idx: {'keys': tensor, 'values': tensor, 'token_indices': []}}}
        self.kv_cache = {}
        
        # Track which pages are on-chip vs off-chip (cloud or disk)
        self.onchip_pages = set()  # Set of (layer_idx, page_idx) tuples
        self.offchip_pages = set()  # Set of (layer_idx, page_idx) tuples that were offloaded
        
        # Page metadata
        self.page_metadata = {}  # {(layer_idx, page_idx): {'last_accessed': timestamp, 'compressed': bool}}
        
        # LRU tracking for page eviction
        self.page_access_history = []  # List of (layer_idx, page_idx) tuples in access order
        
        # Statistics for monitoring
        self.stats = {
            "total_pages": 0,
            "compressed_pages": 0,
            "evicted_pages": 0,
            "offloaded_pages": 0,
            "pruned_tokens": 0,
            "memory_savings_mb": 0.0
        }
        
    def initialize_kv_cache(self, model_config):
        """
        Initialize the key-value cache based on model configuration.
        
        Args:
            model_config: Configuration of the Transformer model
        """
        # Reset data structures
        self.kv_cache = {}
        self.onchip_pages = set()
        self.offchip_pages = set()
        self.page_access_history = []
        self.page_metadata = {}
        
        # Extract model dimensions for memory calculations
        num_layers = getattr(model_config, "num_hidden_layers", 12)
        hidden_size = getattr(model_config, "hidden_size", 768)
        num_attention_heads = getattr(model_config, "num_attention_heads", 12)
        
        # Pre-initialize empty structure for each layer
        for layer_idx in range(num_layers):
            self.kv_cache[layer_idx] = {}
            
        print(f"Initialized KV cache for {num_layers} layers with page size {self.page_size}")
        print(f"Memory budget: {self.max_memory_mb} MB, threshold: {self.memory_threshold_percent}%")
        
    def add_to_kv_cache(self, layer_idx, key, value, token_indices=None):
        """
        Add new key-value pairs to the cache.
        
        Args:
            layer_idx: Index of the Transformer layer
            key: Attention key tensor
            value: Attention value tensor
            token_indices: Indices of tokens these K/V pairs correspond to
        """
        if layer_idx not in self.kv_cache:
            self.kv_cache[layer_idx] = {}
            
        # Determine which page this belongs to based on token positions
        # If token_indices not provided, assume sequential tokens
        if token_indices is None:
            # For autoregressive generation, this would be the latest token
            if HAS_TORCH and isinstance(key, torch.Tensor):
                token_indices = list(range(key.shape[1]))  # Assuming shape [batch, seq_len, ...]
            else:
                token_indices = list(range(key.shape[1] if len(key.shape) > 1 else 1))
        
        # Get sequence length from key tensor
        if HAS_TORCH and isinstance(key, torch.Tensor):
            seq_len = key.shape[1]
        else:
            seq_len = key.shape[1] if len(key.shape) > 1 else 1
            
        # Handle the case where token_indices don't match the sequence length
        # For example, when we have a tensor with seq_len=1 but token_indices=[50]
        # In this case, we want to map the token index 50 to position 0 in our tensor
        if len(token_indices) != seq_len:
            # We're handling a mismatch - likely adding specific tokens
            for token_idx in token_indices:
                page_idx = token_idx // self.page_size
                token_in_page_idx = token_idx % self.page_size
                
                # Create page if it doesn't exist
                if page_idx not in self.kv_cache[layer_idx]:
                    self._allocate_page(layer_idx, page_idx, key.shape, value.shape)
                    
                # Add key/value to the page
                if HAS_TORCH and isinstance(key, torch.Tensor):
                    # For PyTorch tensors, use index selection
                    # If seq_len=1, we map the token to position 0
                    seq_pos = 0 if seq_len == 1 else token_idx % seq_len
                    
                    if seq_pos < seq_len:
                        self.kv_cache[layer_idx][page_idx]['keys'][:, token_in_page_idx] = key.select(1, seq_pos)
                        self.kv_cache[layer_idx][page_idx]['values'][:, token_in_page_idx] = value.select(1, seq_pos)
                else:
                    # Handle numpy arrays or other tensor types
                    seq_pos = 0 if seq_len == 1 else token_idx % seq_len
                    
                    if seq_pos < seq_len:
                        self.kv_cache[layer_idx][page_idx]['keys'][:, token_in_page_idx] = key[:, seq_pos]
                        self.kv_cache[layer_idx][page_idx]['values'][:, token_in_page_idx] = value[:, seq_pos]
                
                # Update token index mapping
                if token_idx not in self.kv_cache[layer_idx][page_idx]['token_indices']:
                    self.kv_cache[layer_idx][page_idx]['token_indices'].append(token_idx)
                    
                # Mark page as recently accessed
                self._mark_page_accessed(layer_idx, page_idx)
        else:
            # Regular case: each position in the tensor corresponds to a token index
            for i, token_idx in enumerate(token_indices):
                page_idx = token_idx // self.page_size
                token_in_page_idx = token_idx % self.page_size
                
                # Create page if it doesn't exist
                if page_idx not in self.kv_cache[layer_idx]:
                    self._allocate_page(layer_idx, page_idx, key.shape, value.shape)
                    
                # Add key/value to the page
                if HAS_TORCH and isinstance(key, torch.Tensor):
                    # For PyTorch tensors, use index selection
                    # Get the corresponding position in the sequence dimension
                    seq_pos = i  # Index in the provided key/value tensors
                    
                    # Make sure we don't go out of bounds
                    if seq_pos < seq_len:
                        self.kv_cache[layer_idx][page_idx]['keys'][:, token_in_page_idx] = key.select(1, seq_pos)
                        self.kv_cache[layer_idx][page_idx]['values'][:, token_in_page_idx] = value.select(1, seq_pos)
                else:
                    # Handle numpy arrays or other tensor types
                    seq_pos = i
                    if seq_pos < seq_len:
                        self.kv_cache[layer_idx][page_idx]['keys'][:, token_in_page_idx] = key[:, seq_pos]
                        self.kv_cache[layer_idx][page_idx]['values'][:, token_in_page_idx] = value[:, seq_pos]
                    
                # Update token index mapping
                if token_idx not in self.kv_cache[layer_idx][page_idx]['token_indices']:
                    self.kv_cache[layer_idx][page_idx]['token_indices'].append(token_idx)
                    
                # Mark page as recently accessed
                self._mark_page_accessed(layer_idx, page_idx)
            
        # Check if we need to evict pages to stay under memory threshold
        self.check_memory_usage_and_evict_if_needed()
        
    def _allocate_page(self, layer_idx, page_idx, key_shape, value_shape):
        """
        Allocate a new page in the KV cache.
        
        Args:
            layer_idx: Index of the Transformer layer
            page_idx: Index of the page to allocate
            key_shape: Shape of key tensors
            value_shape: Shape of value tensors
            
        Returns:
            Newly allocated page
        """
        # Create tensors for this page
        # For keys: [batch, heads, head_dim] -> [batch, page_size, heads, head_dim]
        # For values: similar to keys
        
        if HAS_TORCH and (isinstance(key_shape, torch.Size) or isinstance(value_shape, torch.Size)):
            # PyTorch implementation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.quantization_enabled else torch.float32
            
            # Reshape for page storage: insert page_size dimension
            key_page_shape = list(key_shape)
            key_page_shape[1] = self.page_size  # Replace sequence dimension with page_size
            
            value_page_shape = list(value_shape)
            value_page_shape[1] = self.page_size  # Replace sequence dimension with page_size
            
            # Create empty tensors
            keys = torch.zeros(key_page_shape, dtype=dtype, device=device)
            values = torch.zeros(value_page_shape, dtype=dtype, device=device)
        else:
            # NumPy implementation
            dtype = np.float16 if self.quantization_enabled else np.float32
            
            # Reshape for page storage
            key_page_shape = list(key_shape)
            key_page_shape[1] = self.page_size
            
            value_page_shape = list(value_shape)
            value_page_shape[1] = self.page_size
            
            # Create empty arrays
            keys = np.zeros(key_page_shape, dtype=dtype)
            values = np.zeros(value_page_shape, dtype=dtype)
            
        # Create the page entry
        page = {
            'keys': keys,
            'values': values,
            'token_indices': [],
            'compressed': False
        }
        
        # Store the page
        self.kv_cache[layer_idx][page_idx] = page
        
        # Add to on-chip pages
        page_key = (layer_idx, page_idx)
        self.onchip_pages.add(page_key)
        
        # Initialize metadata
        self.page_metadata[page_key] = {
            'last_accessed': time.time(),
            'compressed': False,
            'size_bytes': self._calculate_page_size(keys, values)
        }
        
        # Update stats
        self.stats["total_pages"] += 1
        
        return page
        
    def get_from_kv_cache(self, layer_idx, token_indices):
        """
        Retrieve key-value pairs from the cache.
        
        Args:
            layer_idx: Index of the Transformer layer
            token_indices: Indices of tokens to retrieve
            
        Returns:
            Retrieved key-value tensors
        """
        if layer_idx not in self.kv_cache:
            return None, None
            
        # Group tokens by pages
        pages_to_tokens = {}
        for token_idx in token_indices:
            page_idx = token_idx // self.page_size
            if page_idx in self.kv_cache[layer_idx]:
                if page_idx not in pages_to_tokens:
                    pages_to_tokens[page_idx] = []
                pages_to_tokens[page_idx].append(token_idx)
        
        # Check if any pages need to be brought back from off-chip storage
        for page_idx in pages_to_tokens.keys():
            page_key = (layer_idx, page_idx)
            if page_key in self.offchip_pages:
                self._fetch_page_from_offchip(layer_idx, page_idx)
                
        # Gather key/value pairs from all relevant pages
        all_keys = []
        all_values = []
        
        for page_idx, page_token_indices in pages_to_tokens.items():
            page = self.kv_cache[layer_idx][page_idx]
            
            # Mark page as accessed
            self._mark_page_accessed(layer_idx, page_idx)
            
            # Check if page is compressed and needs dequantization
            page_key = (layer_idx, page_idx)
            is_compressed = self.page_metadata[page_key].get('compressed', False)
            
            # Extract keys and values for requested tokens
            if HAS_TORCH and isinstance(page['keys'], torch.Tensor):
                # PyTorch implementation
                for token_idx in page_token_indices:
                    token_in_page_idx = token_idx % self.page_size
                    # Only select if the token is actually in this page
                    if token_idx in page['token_indices']:
                        # Get the token's key and value
                        if is_compressed:
                            # Dequantize on-the-fly
                            k = page['keys'][:, token_in_page_idx:token_in_page_idx+1].float() * page['k_scale']
                            v = page['values'][:, token_in_page_idx:token_in_page_idx+1].float() * page['v_scale']
                        else:
                            k = page['keys'][:, token_in_page_idx:token_in_page_idx+1]
                            v = page['values'][:, token_in_page_idx:token_in_page_idx+1]
                        
                        all_keys.append(k)
                        all_values.append(v)
            else:
                # NumPy implementation
                for token_idx in page_token_indices:
                    token_in_page_idx = token_idx % self.page_size
                    if token_idx in page['token_indices']:
                        # Get the token's key and value
                        if is_compressed:
                            # Dequantize on-the-fly
                            k = page['keys'][:, token_in_page_idx:token_in_page_idx+1].astype(np.float32) * page['k_scale']
                            v = page['values'][:, token_in_page_idx:token_in_page_idx+1].astype(np.float32) * page['v_scale']
                        else:
                            k = page['keys'][:, token_in_page_idx:token_in_page_idx+1]
                            v = page['values'][:, token_in_page_idx:token_in_page_idx+1]
                            
                        all_keys.append(k)
                        all_values.append(v)
        
        # Concatenate results from all pages
        if all_keys and all_values:
            if HAS_TORCH and isinstance(all_keys[0], torch.Tensor):
                keys = torch.cat(all_keys, dim=1)
                values = torch.cat(all_values, dim=1)
            else:
                keys = np.concatenate(all_keys, axis=1)
                values = np.concatenate(all_values, axis=1)
            return keys, values
        
        # Return None if no tokens were found
        return None, None
        
    def _mark_page_accessed(self, layer_idx, page_idx):
        """
        Mark a page as recently accessed for LRU tracking.
        
        Args:
            layer_idx: Index of the Transformer layer
            page_idx: Index of the page
        """
        page_key = (layer_idx, page_idx)
        
        # Update access timestamp
        if page_key in self.page_metadata:
            self.page_metadata[page_key]['last_accessed'] = time.time()
        
        # Remove previous occurrence from access history if present
        if page_key in self.page_access_history:
            self.page_access_history.remove(page_key)
            
        # Add to the end of access history (most recently used)
        self.page_access_history.append(page_key)
        
    def compress_page(self, layer_idx, page_idx):
        """
        Apply quantization to compress a memory page.
        
        This implements FP16 → INT8 compression as mentioned in the paper.
        
        Args:
            layer_idx: Index of the layer
            page_idx: Index of the page to compress
            
        Returns:
            Amount of memory saved in bytes
        """
        page_key = (layer_idx, page_idx)
        
        # Check if page exists and is not already compressed
        if (layer_idx not in self.kv_cache or 
            page_idx not in self.kv_cache[layer_idx] or
            page_key not in self.page_metadata or
            self.page_metadata[page_key].get('compressed', False)):
            return 0
            
        page = self.kv_cache[layer_idx][page_idx]
        
        # Calculate pre-compression size
        original_size = self.page_metadata[page_key]['size_bytes']
        
        # Apply compression
        if HAS_TORCH and isinstance(page['keys'], torch.Tensor):
            # PyTorch implementation - simulate INT8 quantization
            # In a real implementation, we would use proper quantization like:
            # torch.quantization.quantize_dynamic or torch.qint8 type
            
            # For simulation, we'll just cast to different dtype
            if page['keys'].dtype != torch.int8:
                # Store scaling factors for dequantization
                k_scale = page['keys'].abs().max().item() / 127.0
                v_scale = page['values'].abs().max().item() / 127.0
                
                # Quantize by scaling and rounding to int8 range
                k_int = (page['keys'] / k_scale).round().clamp(-127, 127).to(torch.int8)
                v_int = (page['values'] / v_scale).round().clamp(-127, 127).to(torch.int8)
                
                # Store quantized values and scales
                page['keys'] = k_int
                page['values'] = v_int
                page['k_scale'] = k_scale
                page['v_scale'] = v_scale
        else:
            # NumPy implementation
            if page['keys'].dtype != np.int8:
                # Store scaling factors
                k_scale = np.abs(page['keys']).max() / 127.0
                v_scale = np.abs(page['values']).max() / 127.0
                
                # Quantize
                k_int = np.clip(np.round(page['keys'] / k_scale), -127, 127).astype(np.int8)
                v_int = np.clip(np.round(page['values'] / v_scale), -127, 127).astype(np.int8)
                
                # Store quantized values and scales
                page['keys'] = k_int
                page['values'] = v_int
                page['k_scale'] = k_scale
                page['v_scale'] = v_scale
                
        # Update metadata
        self.page_metadata[page_key]['compressed'] = True
        
        # Calculate compressed size
        compressed_size = self._calculate_page_size(page['keys'], page['values'])
        self.page_metadata[page_key]['size_bytes'] = compressed_size
        
        # Calculate memory saved
        memory_saved = original_size - compressed_size
        
        # Update stats
        self.stats["compressed_pages"] += 1
        self.stats["memory_savings_mb"] += memory_saved / (1024 * 1024)
        
        print(f"Compressed page ({layer_idx}, {page_idx}): saved {memory_saved / 1024:.2f} KB")
        
        return memory_saved
        
    def _calculate_page_size(self, keys, values):
        """
        Calculate the memory size of a page in bytes.
        
        Args:
            keys: Key tensor
            values: Value tensor
            
        Returns:
            Size in bytes
        """
        if HAS_TORCH and isinstance(keys, torch.Tensor):
            # PyTorch implementation
            key_size = keys.element_size() * keys.nelement()
            value_size = values.element_size() * values.nelement()
        else:
            # NumPy implementation
            key_size = keys.itemsize * keys.size
            value_size = values.itemsize * values.size
            
        # Add some overhead for metadata
        overhead = 100  # bytes
        
        return key_size + value_size + overhead
        
    def evict_pages(self, num_pages_to_evict):
        """
        Evict least recently used pages when memory is constrained.
        
        Args:
            num_pages_to_evict: Number of pages to evict
            
        Returns:
            Indices of evicted pages
        """
        if not self.page_access_history:
            return []
            
        evicted_pages = []
        pages_evicted = 0
        
        # Iterate through pages from least recently used to most recently used
        for page_key in self.page_access_history[::-1]:  # Reverse to get LRU first
            if pages_evicted >= num_pages_to_evict:
                break
                
            layer_idx, page_idx = page_key
            
            # Skip if page is already offchip
            if page_key in self.offchip_pages or page_key not in self.onchip_pages:
                continue
                
            # Try to offload to cloud if enabled
            if self.offloading_enabled:
                self._offload_page_to_cloud(layer_idx, page_idx)
                evicted_pages.append(page_key)
                pages_evicted += 1
            else:
                # If offloading not enabled, just evict the page
                if layer_idx in self.kv_cache and page_idx in self.kv_cache[layer_idx]:
                    # Calculate memory freed
                    memory_freed = self.page_metadata[page_key]['size_bytes']
                    
                    # Remove page from cache
                    del self.kv_cache[layer_idx][page_idx]
                    
                    # Update tracking sets
                    self.onchip_pages.remove(page_key)
                    
                    # Add to evicted list
                    evicted_pages.append(page_key)
                    pages_evicted += 1
                    
                    # Update stats
                    self.stats["evicted_pages"] += 1
                    self.stats["memory_savings_mb"] += memory_freed / (1024 * 1024)
                    
                    print(f"Evicted page ({layer_idx}, {page_idx})")
        
        # Update access history to remove evicted pages
        self.page_access_history = [p for p in self.page_access_history if p not in evicted_pages]
        
        return evicted_pages
        
    def check_memory_usage_and_evict_if_needed(self):
        """
        Check if memory usage exceeds threshold and evict pages if needed.
        
        This method is called regularly to ensure we stay under memory budget.
        
        Returns:
            Boolean indicating if eviction or compression was performed
        """
        # Calculate current memory usage
        current_usage_mb = self.calculate_memory_usage()
        threshold_mb = self.max_memory_mb * self.memory_threshold_percent / 100.0
        
        # Check if we're over threshold
        if current_usage_mb <= threshold_mb:
            return False
            
        print(f"Memory usage ({current_usage_mb:.2f} MB) exceeds threshold ({threshold_mb:.2f} MB)")
        
        # First try compression
        if self.quantization_enabled:
            compressed_any = self._compress_uncompressed_pages()
            
            # Check if compression was enough
            if self.calculate_memory_usage() <= threshold_mb:
                return True
                
        # If still over threshold, evict pages
        excess_mb = max(0, current_usage_mb - threshold_mb)
        avg_page_size_mb = self._estimate_average_page_size_mb()
        
        # Estimate number of pages to evict
        pages_to_evict = max(1, int(excess_mb / avg_page_size_mb * 1.5))  # Add 50% margin
        
        # Evict pages
        evicted_pages = self.evict_pages(pages_to_evict)
        
        return len(evicted_pages) > 0
        
    def _compress_uncompressed_pages(self):
        """
        Compress pages that aren't already compressed to save memory.
        
        Returns:
            Number of pages compressed
        """
        if not self.quantization_enabled:
            return 0
            
        compressed_count = 0
        
        # Sort pages by last accessed time (compress oldest first)
        pages_by_age = sorted(
            self.page_metadata.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        # Compress uncompressed pages
        for (layer_idx, page_idx), metadata in pages_by_age:
            # Skip already compressed pages
            if metadata.get('compressed', False):
                continue
                
            # Skip offchip pages
            page_key = (layer_idx, page_idx)
            if page_key in self.offchip_pages or page_key not in self.onchip_pages:
                continue
                
            # Compress the page
            memory_saved = self.compress_page(layer_idx, page_idx)
            
            if memory_saved > 0:
                compressed_count += 1
                
            # Stop if we've compressed a reasonable number of pages
            if compressed_count >= 5:  # Arbitrary limit to avoid compressing too many in one go
                break
                
        return compressed_count
        
    def _estimate_average_page_size_mb(self):
        """
        Estimate the average size of a page in MB.
        
        Returns:
            Average page size in MB
        """
        if not self.page_metadata:
            return 0.1  # Default estimate
            
        total_size = sum(metadata.get('size_bytes', 0) for metadata in self.page_metadata.values())
        avg_size_mb = total_size / (len(self.page_metadata) * 1024 * 1024)
        
        return max(0.01, avg_size_mb)  # Ensure minimum size estimate
        
    def calculate_memory_usage(self):
        """
        Calculate the current memory usage of the KV cache.
        
        Returns:
            Memory usage in MB
        """
        total_bytes = 0
        
        # Sum up all on-chip page sizes
        for page_key in self.onchip_pages:
            if page_key in self.page_metadata:
                total_bytes += self.page_metadata[page_key].get('size_bytes', 0)
                
        # Convert to MB
        total_mb = total_bytes / (1024 * 1024)
        
        return total_mb
        
    def remove_pruned_tokens(self, token_indices):
        """
        Remove pruned tokens from the KV cache to save memory.
        
        Args:
            token_indices: Indices of tokens that have been pruned
            
        Returns:
            Amount of memory saved in bytes
        """
        if not token_indices:
            return 0
            
        total_memory_saved = 0
        pages_affected = set()
        
        # Group token indices by page
        tokens_by_page = {}
        for token_idx in token_indices:
            page_idx = token_idx // self.page_size
            if page_idx not in tokens_by_page:
                tokens_by_page[page_idx] = []
            tokens_by_page[page_idx].append(token_idx)
            
        # Process each layer
        for layer_idx in self.kv_cache:
            for page_idx, page_token_indices in tokens_by_page.items():
                if page_idx in self.kv_cache[layer_idx]:
                    page = self.kv_cache[layer_idx][page_idx]
                    page_token_indices_set = set(page_token_indices)
                    
                    # Find tokens to remove that are actually in this page
                    tokens_to_remove = page_token_indices_set.intersection(set(page['token_indices']))
                    
                    if tokens_to_remove:
                        # Track original size
                        page_key = (layer_idx, page_idx)
                        original_size = self.page_metadata[page_key]['size_bytes']
                        
                        # For each token to remove, zero out its entries
                        for token_idx in tokens_to_remove:
                            token_in_page_idx = token_idx % self.page_size
                            
                            if HAS_TORCH and isinstance(page['keys'], torch.Tensor):
                                # PyTorch implementation
                                page['keys'][:, token_in_page_idx] = 0
                                page['values'][:, token_in_page_idx] = 0
                            else:
                                # NumPy implementation
                                page['keys'][:, token_in_page_idx] = 0
                                page['values'][:, token_in_page_idx] = 0
                                
                            # Remove from token indices
                            if token_idx in page['token_indices']:
                                page['token_indices'].remove(token_idx)
                                
                        # Track which pages were affected
                        pages_affected.add(page_key)
                        
                        # Check if page is now empty
                        if not page['token_indices']:
                            # If page is empty, we can free it entirely
                            memory_saved = original_size
                            
                            # Remove page
                            del self.kv_cache[layer_idx][page_idx]
                            
                            # Update tracking
                            self.onchip_pages.discard(page_key)
                            self.offchip_pages.discard(page_key)
                            
                            # Remove from metadata
                            if page_key in self.page_metadata:
                                del self.page_metadata[page_key]
                                
                            # Remove from access history
                            while page_key in self.page_access_history:
                                self.page_access_history.remove(page_key)
                                
                            # Update stats
                            self.stats["total_pages"] -= 1
                            total_memory_saved += memory_saved
                            print(f"Removed empty page ({layer_idx}, {page_idx}) after pruning")
                            
        # Update stats
        self.stats["pruned_tokens"] += len(token_indices)
        self.stats["memory_savings_mb"] += total_memory_saved / (1024 * 1024)
        
        # If any tokens were pruned, check memory usage
        if pages_affected:
            self.check_memory_usage_and_evict_if_needed()
            
        return total_memory_saved
        
    def _offload_page_to_cloud(self, layer_idx, page_idx):
        """
        Offload a page to the cloud/disk to free local memory.
        
        Args:
            layer_idx: Index of the layer
            page_idx: Index of the page
            
        Returns:
            Amount of memory saved in bytes
        """
        page_key = (layer_idx, page_idx)
        
        # Check if the page exists and is on-chip
        if page_key not in self.onchip_pages or page_key in self.offchip_pages:
            return 0
            
        # In a real implementation, we would serialize and send the page here
        # For simulation, we'll just track that it happened
        
        # Calculate memory saved
        memory_saved = self.page_metadata[page_key]['size_bytes']
        
        # Move from onchip to offchip
        self.onchip_pages.remove(page_key)
        self.offchip_pages.add(page_key)
        
        # Update stats
        self.stats["offloaded_pages"] += 1
        self.stats["memory_savings_mb"] += memory_saved / (1024 * 1024)
        
        print(f"Offloaded page ({layer_idx}, {page_idx}) to cloud")
        
        return memory_saved
        
    def _fetch_page_from_offchip(self, layer_idx, page_idx):
        """
        Fetch a page from off-chip storage back to on-chip memory.
        
        Args:
            layer_idx: Index of the layer
            page_idx: Index of the page
            
        Returns:
            Boolean indicating success
        """
        page_key = (layer_idx, page_idx)
        
        # Check if the page is actually off-chip
        if page_key not in self.offchip_pages:
            return False
            
        # In a real implementation, we would fetch the page from cloud/disk here
        # For simulation, we'll just track that it happened
        
        # Before bringing back, check if we need to evict something else
        current_usage_mb = self.calculate_memory_usage()
        page_size_mb = self.page_metadata[page_key].get('size_bytes', 0) / (1024 * 1024)
        
        if current_usage_mb + page_size_mb > self.max_memory_mb:
            # Need to make room first
            excess_mb = current_usage_mb + page_size_mb - self.max_memory_mb
            avg_page_size_mb = self._estimate_average_page_size_mb()
            pages_to_evict = max(1, int(excess_mb / avg_page_size_mb * 1.2))  # Add 20% margin
            
            # Evict other pages to make room
            self.evict_pages(pages_to_evict)
            
        # Move from offchip to onchip
        self.offchip_pages.remove(page_key)
        self.onchip_pages.add(page_key)
        
        # Mark as accessed
        self._mark_page_accessed(layer_idx, page_idx)
        
        print(f"Fetched page ({layer_idx}, {page_idx}) from cloud to local memory")
        
        return True
        
    def offload_pages_to_cloud(self, page_indices):
        """
        Offload specific pages to the cloud to free local memory.
        
        Args:
            page_indices: List of (layer_idx, page_idx) tuples to offload
            
        Returns:
            Amount of memory saved in bytes
        """
        if not self.offloading_enabled:
            return 0
            
        total_memory_saved = 0
        
        for layer_idx, page_idx in page_indices:
            memory_saved = self._offload_page_to_cloud(layer_idx, page_idx)
            total_memory_saved += memory_saved
            
        return total_memory_saved
        
    def get_stats(self):
        """
        Get statistics about memory usage and management.
        
        Returns:
            Dictionary of statistics
        """
        # Add current memory usage
        stats = dict(self.stats)
        stats["current_memory_usage_mb"] = self.calculate_memory_usage()
        stats["onchip_pages"] = len(self.onchip_pages)
        stats["offchip_pages"] = len(self.offchip_pages)
        
        return stats 