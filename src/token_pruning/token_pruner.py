"""
Runtime Token Pruning for LLM inference.

This module implements token pruning techniques to reduce computation cost
during autoregressive decoding in transformer-based language models.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

class TokenPruner:
    """
    Runtime Token Pruning Manager
    
    This class handles the identification and removal of low-impact tokens
    during auto-regressive decoding to reduce computation costs.
    
    Key features:
    - Computes attention scores to identify "low-impact" tokens
    - Removes these tokens mid-stream to reduce sequence length
    - Maintains a "shadow set" of removed tokens that can be reintroduced if needed
    - Reduces the quadratic cost of attention when sequences grow large
    
    Based on the second technique from the research paper on efficient
    on-device LLM inference, which uses attention-based thresholding and
    a "shadow set" for potential token reintroduction.
    """
    
    def __init__(self, pruning_threshold=0.23, max_shadow_size=100):
        """
        Initialize the Token Pruner.
        
        Args:
            pruning_threshold: Threshold below which tokens are considered low-impact (default: 0.23)
            max_shadow_size: Maximum number of tokens to keep in the shadow set
        """
        self.pruning_threshold = pruning_threshold
        self.max_shadow_size = max_shadow_size
        self.kv_cache_dim = -2 # Dimension for sequence length in KV cache
        self.shadow_set = {}  # Maps token positions to (token, hidden_state) pairs
        self.token_scores = {}  # Maps token positions to importance scores
        self.pruning_count = 0  # Counter for measuring pruning effectiveness
        self.metrics = {
            "tokens_seen": 0,
            "tokens_pruned": 0,
            "tokens_reintroduced": 0,
            "pruning_time": 0.0,
            "pruning_calls": 0,
            "score_time": 0.0,
            "score_calls": 0
        }
        
    def score_tokens(self, attention_scores, token_indices):
        """
        Compute importance scores for tokens based on attention patterns.
        
        Calculates scores based on the attention received by each past token (key) 
        from the current query token.
        
        Args:
            attention_scores: Attention matrices from the model 
                              [batch_size, num_heads, query_length, key_length].
                              Expected query_length=1 for generation.
            token_indices: Indices of the past tokens (keys) to score.
            
        Returns:
            Dictionary mapping token indices to importance scores.
        """
        start_time = time.time()
        self.metrics["score_calls"] += 1
        scores = {}

        if not (HAS_TORCH and isinstance(attention_scores, torch.Tensor)):
            logger.warning("Cannot score tokens: Attention scores are not a valid PyTorch tensor.")
            return scores # Return empty scores
            
        try:
            # Ensure tensor is on CPU and convert to numpy for easier processing
            attention_scores_np = attention_scores.detach().cpu().numpy()
                
            # Get dimensions - B=batch, H=heads, Q=query_len, K=key_len
            B, H, Q, K = attention_scores_np.shape
            
            if Q != 1:
                logger.warning(f"Expected query length (Q) of 1 for scoring, but got {Q}. Using only the first query position.")

            # Calculate importance score for each token index provided
            for idx in token_indices:
                if idx < 0 or idx >= K:
                    # logger.warning(f"Token index {idx} out of bounds for key length {K}. Skipping.")
                    continue # Skip invalid indices
                
                # Get attention received by token idx (key) from the current query (query pos 0)
                # Shape: [batch_size, num_heads]
                attn_received = attention_scores_np[:, :, 0, idx] 
                
                # Calculate score: Mean attention received across heads for the first batch item
                # Could use max or other aggregation. Mean is simple.
                score = np.mean(attn_received[0, :]) # Use first batch item
                scores[idx] = float(score)
            
            # Update token scores dictionary held by the instance
            self.token_scores.update(scores)
        
        except Exception as e:
            logger.error(f"Error during token scoring: {e}", exc_info=True)
            # Return potentially partial scores calculated so far
            return scores

        # Update timing metric
        self.metrics["score_time"] += (time.time() - start_time) * 1000 # Store time in ms
        
        return scores
        
    def identify_prunable_tokens(self):
        """
        Identify tokens that can be pruned based on their scores.
        
        As described in the paper, tokens with importance score Sⱼ < threshold γ
        are candidates for pruning.
        
        Returns:
            List of token indices that can be safely removed
        """
        prunable_tokens = []
        
        # First few tokens (e.g., 0, 1, 2) are often special tokens that should not be pruned
        protected_tokens = 15  
        
        # Sort tokens by score to ensure we prune the lowest scoring ones first
        sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1])
        
        # Print all token scores for debugging
        # logger.debug("\nToken Scores:")
        # for idx, score in sorted_tokens:
        #     logger.debug(f"Token {idx}: {score:.4f}")

        current_seq_len = len(self.token_scores) # Number of tokens actually scored
        
        for idx, score in sorted_tokens:
            # Skip protected tokens at the beginning of the sequence
            if idx < protected_tokens:
                continue
                
            # <<< FIXED: Never prune the very last token in the current sequence >>>
            if idx == current_seq_len - 1:
                # logger.debug(f"Skipping pruning check for last token index {idx}")
                continue
            # ------------------------------------------------------------------

            # If score is below threshold, mark for pruning
            if score < self.pruning_threshold:
                prunable_tokens.append(idx)
                # logger.debug(f"Token {idx} marked for pruning with score {score:.4f} (below threshold {self.pruning_threshold})")
        
        # if not prunable_tokens:
            # logger.debug("No tokens identified for pruning - all scores above threshold or protected")
        # else:
            # logger.debug(f"Identified {len(prunable_tokens)} tokens for pruning")
        
        return prunable_tokens
        
    def prune_tokens(self, tokens, hidden_states):
        """
        [OBSOLETE - Use prune_state for KV cache]
        Remove low-impact tokens from the active sequence.
        
        As described in the paper, we physically remove pruned tokens from 
        computation and store them in a "shadow set" for potential reintroduction.
        
        Args:
            tokens: The current token sequence
            hidden_states: The model's hidden states for these tokens
            
        Returns:
            Pruned token sequence and updated hidden states
        """
        logger.warning("prune_tokens is likely obsolete. Use prune_state for KV cache.")
        # ... (keep existing implementation for reference or potential other uses)
        # ...
        
    def prune_state(self, 
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], 
                    attention_mask: torch.Tensor, 
                    token_indices_to_prune: List[int]
                   ) -> Tuple[Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], torch.Tensor]:
        """
        Prunes the KV cache (past_key_values) and attention mask based on provided token indices.

        Args:
            past_key_values: The current KV cache tuple from the model.
            attention_mask: The current attention mask.
            token_indices_to_prune: List of sequence indices to remove.

        Returns:
            A tuple containing the pruned past_key_values and pruned attention_mask.
        """
        if not token_indices_to_prune:
            # logger.debug("No indices provided to prune_state.")
            return past_key_values, attention_mask

        if past_key_values is None:
             logger.warning("Cannot prune state: past_key_values is None.")
             return None, attention_mask
             
        # <<< ADDED: Log shapes of all incoming layer KVs >>>
        if logger.isEnabledFor(logging.DEBUG):
             logger.debug("--- prune_state: Received KV Cache Shapes ---")
             for layer_idx, (k, v) in enumerate(past_key_values):
                  k_shape = k.shape if k is not None else None
                  v_shape = v.shape if v is not None else None
                  logger.debug(f"  Layer {layer_idx}: Key={k_shape}, Value={v_shape}")
             logger.debug("---------------------------------------------")
        # -----------------------------------------------------

        try:
            start_time = time.time()
            # 1. Get KV Cache Length (from first layer)
            if past_key_values[0] is None or past_key_values[0][0] is None:
                 logger.error("Cannot prune state: First layer KV cache is None.")
                 return past_key_values, attention_mask
            kv_cache_seq_len = past_key_len = past_key_values[0][0].shape[-2]
            # original_seq_len = attention_mask.shape[-1] # Length from mask is one step ahead
            # --------------------------------------------------
            device = attention_mask.device

            # Create a set for efficient lookup
            prune_set = set(token_indices_to_prune)

            # Determine indices to keep *within the KV cache length*
            keep_indices = [i for i in range(kv_cache_seq_len) if i not in prune_set]
            if not keep_indices: 
                logger.warning("Pruning resulted in zero keep_indices for KV cache. Returning original state.")
                return past_key_values, attention_mask
            keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)
            
            # <<< ADDED: Detailed logging before mask pruning >>>
            mask_dim_size = attention_mask.shape[-1]
            logger.debug(f"--- prune_state: Before Mask Pruning --- ")
            logger.debug(f"  Input token_indices_to_prune: {token_indices_to_prune}")
            logger.debug(f"  Input attention_mask.shape: {attention_mask.shape}")
            logger.debug(f"  Calculated kv_cache_seq_len: {kv_cache_seq_len}")
            logger.debug(f"  Calculated keep_indices (list): {keep_indices}")
            logger.debug(f"  Calculated keep_indices_tensor: {keep_indices_tensor}")
            if len(keep_indices_tensor) > 0:
                 min_idx = torch.min(keep_indices_tensor).item()
                 max_idx = torch.max(keep_indices_tensor).item()
                 logger.debug(f"  keep_indices_tensor Min: {min_idx}, Max: {max_idx}")
            else:
                 logger.debug("  keep_indices_tensor is empty")
            logger.debug(f"  Target mask_dim_size: {mask_dim_size}")
            # -------------------------------------------------
            
            # No need to check if keep_indices == kv_cache_seq_len, proceed with pruning mask/KV
            # if len(keep_indices) == kv_cache_seq_len:
            #      logger.debug("Prune indices provided resulted in no change to KV cache length.")
            #      return past_key_values, attention_mask

            # Prune the attention mask carefully - REVISED APPROACH
            original_mask_len = attention_mask.shape[-1]
            
            # Ensure keep_indices_tensor is valid for selection
            keep_indices_tensor_mask = torch.clamp(keep_indices_tensor, 0, original_mask_len - 1)
            
            # Select the parts of the original mask
            # Part 1: Select columns corresponding to keep_indices (up to original_mask_len)
            mask_part1 = torch.index_select(attention_mask, dim=-1, index=keep_indices_tensor_mask)
            
            # Part 2: Keep any trailing mask tokens (e.g., for the token being generated)
            # Determine if there are trailing tokens beyond the KV cache length in the original mask
            if original_mask_len > kv_cache_seq_len:
                mask_part2 = attention_mask[:, kv_cache_seq_len:]
                # Concatenate the selected kept part with the trailing part
                pruned_attention_mask = torch.cat([mask_part1, mask_part2], dim=-1)
            else:
                # If mask length equals KV cache length, the pruned part is the whole mask
                pruned_attention_mask = mask_part1
                
            batch_size = attention_mask.shape[0]
            
            # Loop through each layer in the KV cache
            pruned_kv_cache = []
            # <<< FIXED: Handle None elements within past_key_values >>>
            for layer_idx, layer_kv_pair in enumerate(past_key_values):
                # Check if the layer's KV pair is None (e.g., due to edge-cloud)
                if layer_kv_pair is None:
                    pruned_kv_cache.append(None)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  Layer {layer_idx}: Skipping pruning, KV pair is None.")
                    continue # Move to the next layer
                
                # Unpack the non-None pair
                layer_keys, layer_values = layer_kv_pair
                
                # Also check if the tensors themselves are None (belt and suspenders)
                if layer_keys is None or layer_values is None:
                    pruned_kv_cache.append(None)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  Layer {layer_idx}: Skipping pruning, Key or Value tensor is None.")
                    continue
                    
                # Perform the pruning using index_select
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Layer {layer_idx}: Pruning Key {layer_keys.shape} and Value {layer_values.shape} with {len(keep_indices)} indices.")
                
                try:
                    pruned_keys = torch.index_select(layer_keys, dim=self.kv_cache_dim, index=keep_indices_tensor)
                    pruned_values = torch.index_select(layer_values, dim=self.kv_cache_dim, index=keep_indices_tensor)
                    pruned_kv_cache.append((pruned_keys, pruned_values))
                except Exception as idx_e:
                    logger.error(f"Error during index_select for layer {layer_idx}: {idx_e}", exc_info=True)
                    logger.error(f"  Shapes: Key={layer_keys.shape}, Value={layer_values.shape}, Index={keep_indices_tensor.shape}")
                    # On error, append the original layer state to avoid breaking structure
                    pruned_kv_cache.append(layer_kv_pair)
            
            pruned_past_key_values_tuple = tuple(pruned_kv_cache)
            # Calculate num_pruned based on change in KV cache length
            num_pruned = kv_cache_seq_len - len(keep_indices)
            
            # Update internal metrics if desired (adjust existing metrics or add new ones)
            self.metrics["tokens_pruned"] += num_pruned # Assuming this counts active pruning
            self.metrics["pruning_time"] += (time.time() - start_time) * 1000
            self.metrics["pruning_calls"] += 1

            logger.info(f"Pruned {num_pruned} tokens. New seq len: {len(keep_indices)}")
            return pruned_past_key_values_tuple, pruned_attention_mask
            
        except Exception as e:
            logger.error(f"Error during state pruning: {e}", exc_info=True)
            # Return original state on error to avoid crashing generation
            return past_key_values, attention_mask

    def _add_to_shadow_set(self, tokens, hidden_states, token_indices):
        """
        Add pruned tokens to the shadow set for potential later reintroduction.
        
        This implements the "shadow set" described in the paper.
        
        Args:
            tokens: The current token sequence
            hidden_states: The model's hidden states for these tokens
            token_indices: Indices of tokens to add to shadow set
        """
        # For each token to be pruned
        for idx in token_indices:
            # Store token and its hidden state
            token = tokens[idx]
            
            # If hidden_states is a tensor (e.g., PyTorch)
            if hidden_states is not None:
                if HAS_TORCH and isinstance(hidden_states, torch.Tensor):
                    # Extract and store the hidden state for this token
                    token_hidden = hidden_states[:, idx, :].detach().cpu().numpy()
                else:
                    # Assume numpy array or similar
                    token_hidden = hidden_states[:, idx, :]
            else:
                token_hidden = None
                
            # Add to shadow set with position information
            self.shadow_set[idx] = (token, token_hidden, self.token_scores.get(idx, 0.0))
            
        # Limit shadow set size if needed
        if len(self.shadow_set) > self.max_shadow_size:
            # Remove tokens with lowest scores first
            sorted_items = sorted(self.shadow_set.items(), key=lambda x: x[1][2])
            excess = len(self.shadow_set) - self.max_shadow_size
            for i in range(excess):
                idx = sorted_items[i][0]
                del self.shadow_set[idx]
        
    def should_reintroduce_tokens(self, current_context):
        """
        Determine if any tokens from the shadow set should be reintroduced.
        
        Args:
            current_context: The current active token context
            
        Returns:
            Boolean indicating if token reintroduction is needed
        """
        if not self.shadow_set:
            return False
            
        # Simple implementation: if the context is focusing on a specific topic,
        # check if shadow tokens might be relevant
        
        # Get last few tokens as the current focus
        last_n = 5
        recent_tokens = current_context[-last_n:] if len(current_context) >= last_n else current_context
        
        # For a real implementation, we would analyze recent attention patterns to see
        # if any pruned tokens might be relevant to the current context
        
        # Simplified heuristic: If sequence is getting long again, consider reintroducing
        # tokens that had higher scores
        if len(current_context) > 50:  # Arbitrary threshold
            high_scoring_shadows = [idx for idx, (_, _, score) in self.shadow_set.items() 
                                    if score > self.pruning_threshold / 2]
            return len(high_scoring_shadows) > 0
            
        return False
        
    def reintroduce_tokens(self, current_tokens, current_states):
        """
        Reintroduce tokens from the shadow set into the active sequence.
        
        As described in the paper, tokens from the shadow set can be reintroduced
        if they become relevant again.
        
        Args:
            current_tokens: The current token sequence
            current_states: The current hidden states
            
        Returns:
            Updated token sequence and hidden states with reintroduced tokens
        """
        if not self.shadow_set or not self.should_reintroduce_tokens(current_tokens):
            return current_tokens, current_states
            
        # Get shadow tokens with scores above half the pruning threshold
        candidate_indices = [idx for idx, (_, _, score) in self.shadow_set.items() 
                             if score > self.pruning_threshold / 2]
        
        if not candidate_indices:
            return current_tokens, current_states
            
        # Sort by score (highest first)
        candidate_indices.sort(key=lambda idx: self.shadow_set[idx][2], reverse=True)
        
        # Limit number of tokens to reintroduce
        max_reintroduce = 5  # Arbitrary limit
        candidate_indices = candidate_indices[:max_reintroduce]
        
        # Prepare for reintroduction
        reintroduced_tokens = list(current_tokens)
        reintroduction_count = 0
        
        if current_states is not None:
            # For PyTorch tensors
            if HAS_TORCH and isinstance(current_states, torch.Tensor):
                device = current_states.device
                batch_size, seq_len, hidden_dim = current_states.shape
                
                # For simplicity, insert shadow tokens at the beginning of the sequence
                # In a real implementation, we would insert at appropriate positions
                insert_position = 1  # After first token
                
                for idx in candidate_indices:
                    token, hidden_state, _ = self.shadow_set[idx]
                    
                    # Insert token
                    reintroduced_tokens.insert(insert_position, token)
                    
                    if hidden_state is not None:
                        # Convert numpy array back to tensor if needed
                        hidden_tensor = torch.tensor(hidden_state, device=device)
                        # Make sure dimensions match
                        hidden_tensor = hidden_tensor.reshape(batch_size, 1, hidden_dim)
                        
                        # Create a new tensor with the reintroduced token's hidden state
                        new_states = torch.cat([
                            current_states[:, :insert_position, :],
                            hidden_tensor,
                            current_states[:, insert_position:, :]
                        ], dim=1)
                        
                        current_states = new_states
                    
                    # Remove from shadow set
                    del self.shadow_set[idx]
                    reintroduction_count += 1
                    insert_position += 1  # Move insertion point for next token
            else:
                # For numpy arrays
                if isinstance(current_states, np.ndarray):
                    batch_size, seq_len, hidden_dim = current_states.shape
                    insert_position = 1  # After first token
                    
                    for idx in candidate_indices:
                        token, hidden_state, _ = self.shadow_set[idx]
                        
                        # Insert token
                        reintroduced_tokens.insert(insert_position, token)
                        
                        if hidden_state is not None:
                            # Make sure dimensions match
                            hidden_state = np.reshape(hidden_state, (batch_size, 1, hidden_dim))
                            
                            # Create a new array with the reintroduced token's hidden state
                            new_states = np.concatenate([
                                current_states[:, :insert_position, :],
                                hidden_state,
                                current_states[:, insert_position:, :]
                            ], axis=1)
                            
                            current_states = new_states
                        
                        # Remove from shadow set
                        del self.shadow_set[idx]
                        reintroduction_count += 1
                        insert_position += 1  # Move insertion point for next token
        
        # Update metrics
        self.metrics["tokens_reintroduced"] += reintroduction_count
        
        return reintroduced_tokens, current_states

    def get_pruning_stats(self):
        """
        Get statistics about the pruning process.
        
        Returns:
            Dictionary with pruning statistics
        """
        stats = {
            "total_tokens_seen": self.metrics["tokens_seen"],
            "total_tokens_pruned": self.metrics["tokens_pruned"],
            "total_tokens_reintroduced": self.metrics["tokens_reintroduced"],
            "pruning_ratio": self.metrics["tokens_pruned"] / max(1, self.metrics["tokens_seen"]),
            "shadow_set_size": len(self.shadow_set),
            "avg_pruning_time_ms": (self.metrics["pruning_time"] * 1000) / max(1, self.metrics["pruning_calls"]),
            "avg_scoring_time_ms": (self.metrics["score_time"] * 1000) / max(1, self.metrics["score_calls"])
        }
        return stats
        
    def reset_metrics(self):
        """Reset all metrics to zero."""
        self.metrics = {k: 0 for k in self.metrics}
        self.pruning_count = 0 