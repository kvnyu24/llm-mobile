import torch
import time
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Set
from collections import OrderedDict, deque

# Check if torch is available
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

logger = logging.getLogger("memory_manager")


class MemoryManager:
    """
    Manages the Key-Value (KV) cache for large language models, implementing strategies
    like layer-based quantization (compression) to stay within memory limits when
    working with the real KV cache tuple produced by manual generation loops.
    
    Key features:
    - Tracks real KV cache state (`past_key_values`).
    - Calculates memory usage based on actual tensor sizes.
    - Implements layer-level INT8 quantization when memory threshold is exceeded.
    - Provides mechanism to dequantize layers before use.

    Note: Page-based logic (eviction, paging, offloading) from previous versions
          is currently disabled/obsolete in this layer-focused implementation but
          kept in comments for potential future adaptation.
    """
    
    def __init__(self, max_memory_mb=512, page_size=16, quantization_enabled=True, 
                 memory_threshold_percent=90, offloading_enabled=True, enable_logging=True):
        self.max_memory_mb = max_memory_mb
        self.page_size = page_size # Keep for potential future use (e.g., LRU tracking)
        self.quantization_enabled = quantization_enabled
        self.memory_threshold_percent = memory_threshold_percent
        self.offloading_enabled = offloading_enabled # Not fully implemented for real cache yet
        self.enable_logging = enable_logging
        
        # --- Core State for Real KV Cache --- 
        self.model_config = None
        self.num_layers = 0
        self.hidden_size = 0
        self.num_attention_heads = 0
        self.bytes_per_element = 2 # Estimated from config
        self.real_past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        self.current_seq_len = 0
        # Metadata tracks layer-level state (e.g., compression)
        self.layer_metadata: Dict[int, Dict[str, Any]] = {} 
        # --------------------------------------
        
        # Statistics for monitoring
        self.stats = self._reset_stats()

        logger.info(f"Memory Manager initialized with {max_memory_mb}MB budget, threshold {memory_threshold_percent}%, Quantization: {quantization_enabled}")

    def _reset_stats(self) -> Dict[str, Any]:
        """Resets the statistics dictionary."""
        return {
            "compressions": 0,          # Layer compressions triggered (alias for quantization)
            "compression_time_ms": 0.0,
            "memory_savings_mb": 0.0,
            "dequantization_time_ms": 0.0,
            "estimated_memory_usage_mb": 0.0, # Current estimate
            "peak_memory_usage_mb": 0.0, 
            "quantization_events": 0,   
            "quantized_layers": set(),   
            "dequantization_events": 0, 
            #  Eviction Stats
            "eviction_events": 0,
            "evicted_layers": set(),
            "memory_saved_by_eviction_mb": 0.0,
            # ---------------------------
            # --- Obsolete/Unused Stats ---
            # "allocations": 0,
            # "evictions": 0,
            # "compressed_pages": 0,
            # "offloaded_pages": 0,
            # "fetched_pages": 0,
            # "cache_hits": 0,
            # "cache_misses": 0,
            # "pruned_tokens": 0,
            # "eviction_time_ms": 0.0,
            # "offload_time_ms": 0.0,
            # "fetch_time_ms": 0.0,
        }

    def initialize_kv_cache(self, model_config):
        """Resets the manager state and stores model configuration."""
        # Reset core state
        self.real_past_key_values = None
        self.current_seq_len = 0
        self.layer_metadata = {}
        self.stats = self._reset_stats()

        # Store model config
        self.model_config = model_config
        # Safely get attributes, providing defaults if needed
        self.num_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", 0))
        self.hidden_size = getattr(model_config, "hidden_size", getattr(model_config, "n_embd", 0))
        self.num_attention_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "n_head", 0))
        
        # Check if essential config values were found
        if self.num_layers == 0 or self.hidden_size == 0 or self.num_attention_heads == 0:
             logger.warning("Could not determine all model dimensions (num_layers, hidden_size, num_heads) from config. Memory calculations might be inaccurate.")

        # Estimate bytes per element based on reported dtype if available
        model_dtype = getattr(model_config, "torch_dtype", None)
        if HAS_TORCH:
            if model_dtype == torch.float32:
                self.bytes_per_element = 4
            elif model_dtype == torch.float16 or model_dtype == torch.bfloat16:
                self.bytes_per_element = 2
            else:
                self.bytes_per_element = 2 # Default guess if dtype unknown or not torch
                if model_dtype:
                     logger.warning(f"Unknown torch_dtype {model_dtype} in config, assuming 2 bytes/element.")
        else:
             self.bytes_per_element = 2 # Default guess if no torch
            
        if self.enable_logging:
            logger.info(f"Initialized Memory Manager state for {self.num_layers} layers (config: {self.hidden_size} hidden, {self.num_attention_heads} heads, {self.bytes_per_element} bytes/element estimate)")
            logger.info(f"Memory budget: {self.max_memory_mb} MB, threshold: {self.memory_threshold_percent}%")
        
    def update_state(self, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], current_seq_len: int):
        """Update the manager with the current real past_key_values state."""
        self.real_past_key_values = past_key_values
        self.current_seq_len = current_seq_len
        # Calculate and store current usage immediately
        current_usage_mb = self.calculate_memory_usage()
        if self.enable_logging:
            logger.debug(f"Updated real KV cache state. Seq Len: {current_seq_len}. Calculated Usage: {current_usage_mb:.2f} MB")

    def check_memory_usage_and_trigger_actions(self):
        """Checks memory usage and triggers layer compression if needed."""
        if not self.quantization_enabled: # Only action currently is compression
            return True 

        # <<< Use estimated usage from update_state for checks
        usage_before_action_mb = self.stats.get("estimated_memory_usage_mb", 0.0)
        threshold_mb = self.max_memory_mb * (self.memory_threshold_percent / 100.0)

        if self.enable_logging:
             logger.info(f"Checking memory: Usage={usage_before_action_mb:.4f}MB, Threshold={threshold_mb:.2f}MB (Budget={self.max_memory_mb}MB)")

        memory_to_free_mb = usage_before_action_mb - threshold_mb
        if memory_to_free_mb > 0:
            if self.enable_logging:
                 logger.info(f"Memory usage ({usage_before_action_mb:.2f}MB) still exceeds threshold ({threshold_mb:.2f}MB) after compression. Triggering LAYER COMPRESSION.")

            # --- Trigger layer compression --- 
            if self.enable_logging:
                logger.debug("  Entering compression loop...")
            freed_memory_mb = 0
            compression_start_time = time.time()
            compressed_count = 0
            if self.real_past_key_values: # Ensure cache exists
                for layer_idx in range(len(self.real_past_key_values) - 1, -1, -1):
                    if self.enable_logging:
                        logger.debug(f"    Considering layer {layer_idx} for compression.")
                        
                    if freed_memory_mb >= memory_to_free_mb:
                        if self.enable_logging:
                            logger.debug(f"    Freed enough memory ({freed_memory_mb:.2f} >= {memory_to_free_mb:.2f}). Stopping compression.")
                        break # Freed enough memory

                    # Check if layer is already compressed
                    if not self.layer_metadata.get(layer_idx, {}).get('compressed', False):
                        memory_saved_bytes = self._compress_layer(layer_idx)
                        if self.enable_logging:
                            logger.debug(f"    _compress_layer({layer_idx}) returned {memory_saved_bytes} bytes saved.")
                        if memory_saved_bytes > 0:
                            freed_memory_mb += memory_saved_bytes / (1024 * 1024)
                            compressed_count += 1
                    else:
                        if self.enable_logging:
                             logger.debug(f"    Layer {layer_idx} already compressed. Skipping.")
            
            total_compression_time = (time.time() - compression_start_time) * 1000
            self.stats["compression_time_ms"] += total_compression_time # Accumulate time

            if compressed_count > 0:
                 if self.enable_logging:
                    logger.info(f"Layer compression finished. Compressed {compressed_count} layers. Time: {total_compression_time:.2f}ms. Freed: {freed_memory_mb:.2f}MB")
            else:
                 if self.enable_logging:
                    logger.info(f"Layer compression finished. No layers compressed. Time: {total_compression_time:.2f}ms.")
            # -----------------------------------------
        
        # --- Calculate final usage and update peak AFTER actions ---
        usage_after_compression_mb = self.calculate_memory_usage()
        if self.enable_logging:
            logger.debug(f"Memory usage after compression check/actions: {usage_after_compression_mb:.4f}MB")

        # --- Eviction Logic --- 
        final_usage_mb = usage_after_compression_mb # Start with usage after compression
        memory_still_over_mb = final_usage_mb - threshold_mb

        if memory_still_over_mb > 0:
            if self.enable_logging:
                 logger.info(f"Memory usage ({final_usage_mb:.2f}MB) still exceeds threshold ({threshold_mb:.2f}MB) after compression. Triggering EVICTION.")
                 
            eviction_start_time = time.time()
            memory_freed_by_eviction_mb = 0.0
            evicted_count = 0
            
            # Simple strategy: Evict starting from layer 0
            if self.real_past_key_values is not None:
                 # Need to work with a list copy to modify
                 kv_list = list(self.real_past_key_values)
                 
                 for layer_idx in range(len(kv_list)):
                     if final_usage_mb <= threshold_mb:
                          break # Stop evicting if below threshold
                          
                     metadata = self.layer_metadata.get(layer_idx, {})
                     # Skip if already evicted
                     if metadata.get('evicted', False):
                          continue 
                     
                     current_layer_kv = kv_list[layer_idx]
                     bytes_to_free = 0
                     
                     # Calculate bytes used by this layer (compressed or not)
                     if metadata.get('compressed', False):
                          keys_quant = metadata.get('quantized_keys')
                          values_quant = metadata.get('quantized_values')
                          if HAS_TORCH and isinstance(keys_quant, torch.Tensor) and isinstance(values_quant, torch.Tensor):
                               bytes_to_free = (keys_quant.nelement() + values_quant.nelement()) * 1
                     elif isinstance(current_layer_kv, tuple) and len(current_layer_kv) == 2 and current_layer_kv != (None, None):
                          bytes_to_free = self._calculate_tensor_pair_size(current_layer_kv[0], current_layer_kv[1])
                     
                     if bytes_to_free > 0:
                          # Evict the layer
                          if self.enable_logging:
                               logger.info(f"Evicting Layer {layer_idx} (saving {bytes_to_free / (1024*1024):.2f} MB)")
                          kv_list[layer_idx] = (None, None) # Set KV to None
                          
                          # Update metadata
                          metadata['evicted'] = True
                          metadata['compressed'] = False # Cannot be compressed if evicted
                          metadata.pop('quantized_keys', None)
                          metadata.pop('quantized_values', None)
                          metadata.pop('key_scale', None)
                          # ... remove other compression metadata ...
                          self.layer_metadata[layer_idx] = metadata # Store updated metadata
                          
                          # Update stats
                          self.stats["eviction_events"] += 1
                          self.stats["evicted_layers"].add(layer_idx)
                          mb_freed = bytes_to_free / (1024 * 1024)
                          self.stats["memory_saved_by_eviction_mb"] += mb_freed
                          memory_freed_by_eviction_mb += mb_freed
                          evicted_count += 1
                          
                          # Update current usage estimate
                          final_usage_mb -= mb_freed
                 
                 # Convert list back to tuple
                 self.real_past_key_values = tuple(kv_list)
                 
            total_eviction_time_ms = (time.time() - eviction_start_time) * 1000
            if self.enable_logging:
                logger.info(f"Eviction finished. Evicted {evicted_count} layers. Time: {total_eviction_time_ms:.2f}ms. Freed: {memory_freed_by_eviction_mb:.2f}MB")
                logger.info(f"Memory usage after eviction: {final_usage_mb:.4f}MB")
        # -----------------------------
            
        # Update peak based on potentially reduced size (after compression AND eviction)
        if final_usage_mb > self.stats.get("peak_memory_usage_mb", 0.0):
            self.stats["peak_memory_usage_mb"] = final_usage_mb
            if self.enable_logging:
                 logger.debug(f"Updated peak memory usage to: {final_usage_mb:.4f}MB")
        elif final_usage_mb < self.stats.get("peak_memory_usage_mb", 0.0):
             # Peak doesn't decrease, but we might log the difference for clarity
             if self.enable_logging:
                 logger.debug(f"Memory usage ({final_usage_mb:.4f}MB) is lower than peak ({self.stats.get('peak_memory_usage_mb', 0.0):.4f}MB) after actions.")

        # Return status based on whether final usage is below threshold
        return final_usage_mb <= threshold_mb

    def _compress_layer(self, layer_idx: int) -> int:
        """
        Compress an entire layer's K/V tensors using INT8 quantization.
        Modifies the tensors within self.real_past_key_values in-place.
        
        Args:
            layer_idx: Index of the Transformer layer to compress.
            
        Returns:
            Memory saved in bytes, or 0 if not compressed or error.
        """
        if not self.quantization_enabled or not HAS_TORCH:
            return 0

        if not self.real_past_key_values or layer_idx >= len(self.real_past_key_values):
            logger.warning(f"Cannot compress layer {layer_idx}: Invalid past_key_values state.")
            return 0

        # --- Check if layer entry is None before unpacking --- 
        layer_entry = self.real_past_key_values[layer_idx]
        if layer_entry is None:
            logger.debug(f"Skipping compression for layer {layer_idx}: KV entry is None.")
            return 0
        # ------------------------------------------------------
            
        try:
            # Attempt to unpack now that we know it's not None
            keys, values = layer_entry

            if keys is None or values is None:
                 # logger.debug(f"Layer {layer_idx} has no KV cache to compress.")
                 return 0

            # Ensure tensors are on CPU for quantization (if not already)
            # This might be slow if tensors are large and on GPU
            # Consider performing quantization directly on GPU if possible
            original_device = keys.device
            keys = keys.cpu()
            values = values.cpu()
            
            # --- Quantization --- 
            original_key_size = keys.element_size() * keys.nelement()
            original_value_size = values.element_size() * values.nelement()
            original_total_bytes = original_key_size + original_value_size
            
            # Calculate scale and zero point for keys
            k_min, k_max = keys.min(), keys.max()
            k_scale = (k_max - k_min) / 255.0
            k_zero_point = torch.round(-k_min / k_scale) # Use round for symmetry
            k_zero_point = torch.clamp(k_zero_point, 0, 255).to(torch.uint8)
            keys_quant = torch.clamp(torch.round(keys / k_scale + k_zero_point), 0, 255).to(torch.uint8)
            
            # Calculate scale and zero point for values
            v_min, v_max = values.min(), values.max()
            v_scale = (v_max - v_min) / 255.0
            v_zero_point = torch.round(-v_min / v_scale)
            v_zero_point = torch.clamp(v_zero_point, 0, 255).to(torch.uint8)
            values_quant = torch.clamp(torch.round(values / v_scale + v_zero_point), 0, 255).to(torch.uint8)
            # ---------------------
            
            compressed_key_size = keys_quant.element_size() * keys_quant.nelement()
            compressed_value_size = values_quant.element_size() * values_quant.nelement()
            compressed_total_bytes = compressed_key_size + compressed_value_size
            
            memory_saved = original_total_bytes - compressed_total_bytes
            
            # Store quantized tensors and metadata
            # Instead of replacing in self.real_past_key_values (which expects original dtype),
            # we store the quantized data in metadata. The real cache tensor becomes None.
            self.layer_metadata[layer_idx] = {
                'compressed': True,
                'quantized_keys': keys_quant,
                'key_scale': k_scale,
                'key_zero_point': k_zero_point,
                'quantized_values': values_quant,
                'value_scale': v_scale,
                'value_zero_point': v_zero_point,
                'original_dtype': keys.dtype, # Store original dtype
                'original_device': original_device # Store original device
            }
            
            # Nullify the original tensor pair in the tuple to reflect freed memory
            # Note: This modifies the tuple structure; ensure callers handle None
            # We can't directly modify the tuple, so we have to rebuild it if needed,
            # but for now, we assume the `calculate_memory_usage` checks metadata.
            # self.real_past_key_values[layer_idx] = (None, None) # This doesn't work on tuples
            # Let calculate_memory_usage handle the logic based on metadata['compressed']
            
            # Update states
            self.stats["compressions"] += 1
            self.stats["quantization_events"] += 1
            self.stats["quantized_layers"].add(layer_idx)
            self.stats["memory_savings_mb"] += memory_saved / (1024 * 1024)
            # --------------------------
        
            if self.enable_logging:
                    size_kb = memory_saved / 1024
                    logger.info(f"Compressed Layer {layer_idx} [INT8]. Saved {size_kb:.2f} KB. Time: {time.time() * 1000:.2f} ms.") # Log time inside?
            
            return memory_saved
        except Exception as e:
             logger.error(f"Error during INT8 quantization for layer {layer_idx}: {e}", exc_info=True)
             # Ensure metadata reflects failure
             self.layer_metadata[layer_idx] = self.layer_metadata.get(layer_idx, {}) 
             self.layer_metadata[layer_idx]['compressed'] = False
             return 0

    def _dequantize_layer(self, layer_idx: int):
        """
        Dequantizes a layer if it was previously compressed.
        Restores the K/V tensors to their original dtype and device.
        This is called *before* a layer is used in the forward pass.
        """
        metadata = self.layer_metadata.get(layer_idx)
        if not (metadata and metadata.get('compressed', False)):
             # logger.debug(f"Layer {layer_idx} not compressed or no metadata, skipping dequantization.")
             return
             
        # Check if evicted
        if metadata.get('evicted', False):
             if self.enable_logging:
                  logger.warning(f"Attempted to dequantize layer {layer_idx}, but it is marked as evicted.")
             return
        # ---------------------------
            
        if not HAS_TORCH:
            logger.warning("Torch not available, cannot dequantize layer {layer_idx}")
            return
        
        try:
            start_time = time.time()
            
            keys_quant = metadata['quantized_keys']
            values_quant = metadata['quantized_values']
            k_scale = metadata['key_scale']
            k_zero_point = metadata['key_zero_point']
            v_scale = metadata['value_scale']
            v_zero_point = metadata['value_zero_point']
            original_dtype = metadata['original_dtype']
            original_device = metadata['original_device']
            
            # Dequantize
            keys_dequant = (keys_quant.float() - k_zero_point.float()) * k_scale
            values_dequant = (values_quant.float() - v_zero_point.float()) * v_scale
            
            # Cast back to original dtype and move to original device
            keys_restored = keys_dequant.to(dtype=original_dtype, device=original_device)
            values_restored = values_dequant.to(dtype=original_dtype, device=original_device)
            
            # Restore the tensor pair in self.real_past_key_values
            # This requires modifying the tuple structure, which isn't ideal.
            # Alternative: Pass the dequantized tensors directly to the layer?
            # For now, let's assume we *can* update the tuple (may need refactoring)
            if self.real_past_key_values is not None and 0 <= layer_idx < len(self.real_past_key_values):
                 # Create a list, modify, then convert back to tuple
                 kv_list = list(self.real_past_key_values)
                 kv_list[layer_idx] = (keys_restored, values_restored)
                 self.real_past_key_values = tuple(kv_list)
            else:
                 logger.warning(f"Could not restore dequantized layer {layer_idx} to real_past_key_values tuple.")

            # Update metadata
            metadata['compressed'] = False
            metadata.pop('quantized_keys', None) # Remove quantized data
            metadata.pop('quantized_values', None)
            
            # Update stats
            self.stats["dequantization_events"] += 1
            dequant_time = (time.time() - start_time) * 1000
            self.stats["dequantization_time_ms"] += dequant_time
            # --------------------------

            if self.enable_logging:
                logger.debug(f"Dequantized layer {layer_idx}. Time: {dequant_time:.2f} ms.")

        except Exception as e:
             logger.error(f"Error during dequantization for layer {layer_idx}: {e}", exc_info=True)
             # Leave metadata as compressed=True if error occurs? Or set to False?
             metadata['compressed'] = True # Keep compressed state on error? safer?

    def dequantize_all_compressed_layers(self):
        """Iterates through all layers and dequantizes any that are marked as compressed."""
        if not (HAS_TORCH and self.real_past_key_values):
             return # Nothing to dequantize

        dequant_count = 0
        needs_dequant = False
        for layer_idx in range(len(self.real_past_key_values)):
             if self.layer_metadata.get(layer_idx, {}).get('compressed', False):
                  needs_dequant = True
                  break # Found at least one compressed layer

        if not needs_dequant:
            return

        if self.enable_logging:
            logger.info("Performing pre-step dequantization of compressed layers...")

        overall_start_time = time.time()
        for layer_idx in range(len(self.real_past_key_values)):
            # Check again inside loop, _dequantize_layer might have failed and reset flag
            if self.layer_metadata.get(layer_idx, {}).get('compressed', False):
                self._dequantize_layer(layer_idx)
                # Check if still compressed after attempt (might fail)
                if not self.layer_metadata.get(layer_idx, {}).get('compressed', False):
                    dequant_count += 1

        if dequant_count > 0:
            overall_dequant_time_ms = (time.time() - overall_start_time) * 1000
            if self.enable_logging:
                logger.info(f"Dequantized {dequant_count} layers. Total time: {overall_dequant_time_ms:.2f} ms.")
            # Note: Individual times are summed in _dequantize_layer into stats["dequantization_time_ms"]

    def _calculate_tensor_pair_size(self, keys: Optional[torch.Tensor], values: Optional[torch.Tensor]) -> int:
        """Calculate the memory size of a K/V tensor pair in bytes."""
        key_bytes = 0
        value_bytes = 0
        if HAS_TORCH:
            if isinstance(keys, torch.Tensor):
                try:
                    # Detailed Logging
                    if self.enable_logging:
                         logger.debug(f"    _calc_size: Key Tensor - Device={keys.device}, Dtype={keys.dtype}, Shape={keys.shape}, NElement={keys.nelement()}, ElemSize={keys.element_size()}")
                    # ----------------------------
                    key_bytes = keys.nelement() * keys.element_size()
                except RuntimeError as e: # Handle potential errors like meta device tensors
                    logger.warning(f"Could not get size for key tensor: {e}")
            if isinstance(values, torch.Tensor):
                 try:
                    # Detailed Logging
                    if self.enable_logging:
                        logger.debug(f"    _calc_size: Value Tensor - Device={values.device}, Dtype={values.dtype}, Shape={values.shape}, NElement={values.nelement()}, ElemSize={values.element_size()}")
                    # ----------------------------
                    value_bytes = values.nelement() * values.element_size()
                 except RuntimeError as e:
                    logger.warning(f"Could not get size for value tensor: {e}")
        # Add numpy fallback if needed later
        # elif isinstance(keys, np.ndarray): key_bytes = keys.nbytes ...
        if self.enable_logging:
             logger.debug(f"    _calc_size: Calculated Bytes = {key_bytes + value_bytes}")
        return key_bytes + value_bytes

    def calculate_memory_usage(self) -> float:
        """
        Calculate the current memory usage by inspecting the real past_key_values tensors.
        Returns:
            Current memory usage in MB.
        """
        if not (HAS_TORCH and self.real_past_key_values):
            self.stats["estimated_memory_usage_mb"] = 0.0
            return 0.0

        total_bytes = 0
        try:
            for layer_idx, layer_kv in enumerate(self.real_past_key_values):
                metadata = self.layer_metadata.get(layer_idx)
                if metadata and metadata.get('compressed', False):
                    # Layer is compressed - calculate size from quantized data
                    keys_quant = metadata.get('quantized_keys')
                    values_quant = metadata.get('quantized_values')
                    if HAS_TORCH and isinstance(keys_quant, torch.Tensor) and isinstance(values_quant, torch.Tensor):
                        # INT8 tensors, element size is 1
                        layer_bytes = (keys_quant.nelement() + values_quant.nelement()) * 1
                        if self.enable_logging:
                             logger.debug(f"    _calc_usage: Layer {layer_idx} (Compressed) Bytes = {layer_bytes}")
                        total_bytes += layer_bytes
                    else:
                         if self.enable_logging:
                              logger.warning(f"Compressed metadata found for layer {layer_idx} but quantized tensors invalid.")
                elif isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                    # Layer is not compressed (or no metadata) - use original tensors
                    key_tensor, value_tensor = layer_kv
                    # Use the existing helper, which logs details
                    total_bytes += self._calculate_tensor_pair_size(key_tensor, value_tensor)
                # else: Handle potential None K/V pairs if necessary
                # Explicitly handle None/Evicted layers
                elif layer_kv is None or layer_kv == (None, None):
                     if self.enable_logging:
                          # Check metadata for why it might be None
                          evicted_flag = metadata.get('evicted', False) if metadata else False
                          logger.debug(f"    _calc_usage: Layer {layer_idx} KV is None (Evicted: {evicted_flag}). Bytes = 0")
                     # Contribute 0 bytes
                     pass 
                # --------------------------------------------------
        except Exception as e:
            logger.error(f"Error calculating memory usage from real_past_key_values: {e}", exc_info=True)
            self.stats["estimated_memory_usage_mb"] = 0.0 # Reset stat on error
            return 0.0 # Return 0 on error

        if self.enable_logging:
             logger.debug(f"    _calc_usage: Total Calculated Bytes = {total_bytes}")
        # --------------------
        total_mb = total_bytes / (1024 * 1024)
        if self.enable_logging:
             logger.debug(f"    _calc_usage: Total Calculated MB = {total_mb}")
        # --------------------
        self.stats["estimated_memory_usage_mb"] = total_mb 
        return total_mb
        
    def get_stats(self) -> Dict[str, Any]:
        """Returns a dictionary containing performance statistics."""
        current_usage = self.calculate_memory_usage() 
        stats_copy = dict(self.stats)
        stats_copy["current_memory_usage_mb"] = current_usage # Ensure key exists and is current
        # Logging
        if self.enable_logging:
            logger.debug(f"MemoryManager get_stats() returning: {stats_copy}")
        # --------------------
        return stats_copy

    # --- Potentially Obsolete/Adaptable Page-based Methods ---
    # These methods operated on the internal self.kv_cache simulation
    # and would need significant adaptation to work with the layer-based
    # management of self.real_past_key_values.

    # def _calculate_page_size(self, keys, values): ... 
    # def get_from_kv_cache(self, layer_idx, token_indices): ...
    # def _mark_page_accessed(self, layer_idx, page_idx): ...
    # def _allocate_page(self, layer_idx, page_idx, key_shape, value_shape): ...
    # def evict_pages(self, num_pages_to_evict, strategy='priority'): ... # Could be adapted for offloading
    # def _find_eviction_candidates(self, num_to_find, strategy): ...
    # def remove_pruned_tokens(self, token_indices): ...
    # def _offload_page_to_cloud(self, layer_idx, page_idx): ...
    # def _fetch_page_from_offchip(self, layer_idx, page_idx, is_prefetch=False): ...
    # def offload_pages_to_cloud(self, page_indices): ...
    # def prefetch_pages(self, current_token_idx, lookahead=5): ...

