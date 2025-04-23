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
                
            # <<< FIXED: Never prune the very last token in the current sequence
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
             
        # Log shapes of all incoming layer KVs
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
            device = attention_mask.device
            prune_set = set(token_indices_to_prune)

            # --- Prune KV Cache (Layer by Layer) --- 
            new_kv_cache_list = []
            for layer_idx, layer_entry in enumerate(past_key_values):
                # --- Check if layer entry is None before processing/unpacking --- 
                if layer_entry is None:
                    logger.debug(f"  Skipping pruning for layer {layer_idx}: KV entry is None.")
                    new_kv_cache_list.append(None) # Preserve the None entry
                    continue
                # -------------------------------------------------------------
                
                # Unpack now that we know it's not None
                k, v = layer_entry
                
                if k is None or v is None:
                    logger.debug(f"  Skipping pruning for layer {layer_idx}: K or V tensor is None.")
                    new_kv_cache_list.append((k, v)) # Append original tuple if tensors are None
                    continue

                # Calculate keep_indices based on *this layer's* K tensor length
                current_k_seq_len = k.shape[self.kv_cache_dim] # Typically dim -2
                keep_indices_for_layer = [i for i in range(current_k_seq_len) if i not in prune_set]
                
                if not keep_indices_for_layer:
                    logger.warning(f"Pruning layer {layer_idx} resulted in zero keep_indices. Keeping original K/V.")
                    new_kv_cache_list.append((k, v))
                    continue
                
                keep_indices_tensor_for_layer = torch.tensor(keep_indices_for_layer, dtype=torch.long, device=device)

                # Prune K and V using the calculated indices for this layer
                pruned_k = torch.index_select(k, dim=self.kv_cache_dim, index=keep_indices_tensor_for_layer)
                pruned_v = torch.index_select(v, dim=self.kv_cache_dim, index=keep_indices_tensor_for_layer)
                new_kv_cache_list.append((pruned_k, pruned_v))

            pruned_past_key_values = tuple(new_kv_cache_list)
            # -----------------------------------------
            
            # --- Prune Attention Mask --- 
            mask_current_len = attention_mask.shape[-1] # Typically dim -1 for attention mask
            # Calculate keep_indices based on the mask's length
            keep_mask_indices = [i for i in range(mask_current_len) if i not in prune_set]
            
            pruned_attention_mask = attention_mask # Default to original if no indices kept
            if keep_mask_indices:
                keep_mask_indices_tensor = torch.tensor(keep_mask_indices, dtype=torch.long, device=device)
                pruned_attention_mask = torch.index_select(attention_mask, dim=-1, index=keep_mask_indices_tensor)
            else:
                logger.warning("Pruning resulted in zero keep_indices for attention mask. Returning original mask.")
            # -----------------------------------------

            end_time = time.time()
            self.metrics["pruning_time"] += (end_time - start_time) * 1000 # ms
            self.metrics["pruning_calls"] += 1
            pruned_count = attention_mask.shape[-1] - pruned_attention_mask.shape[-1]
            self.metrics["tokens_pruned"] += pruned_count
            
            # Log shapes after pruning
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug("--- prune_state: Resulting KV Cache Shapes ---")
                 for layer_idx, (k, v) in enumerate(pruned_past_key_values):
                      k_shape = k.shape if k is not None else None
                      v_shape = v.shape if v is not None else None
                      logger.debug(f"  Layer {layer_idx}: Key={k_shape}, Value={v_shape}")
                 logger.debug(f"--- prune_state: Resulting Attention Mask Shape: {pruned_attention_mask.shape} ---")
                 logger.debug("---------------------------------------------------")

            return pruned_past_key_values, pruned_attention_mask

        except IndexError as ie:
            logger.error(f"IndexError during prune_state: {ie}", exc_info=True)
            # Log relevant shapes just before the error might occur
            logger.error(f"  Attention Mask Shape: {attention_mask.shape}")
            if 'keep_indices_tensor_for_layer' in locals():
                 logger.error(f"  Last keep_indices_tensor_for_layer: {keep_indices_tensor_for_layer}")
            if 'k' in locals():
                 logger.error(f"  Tensor 'k' shape where error might have occurred: {k.shape}")
            # Return original state on error to prevent crash
            return past_key_values, attention_mask
        except Exception as e:
            logger.error(f"Unexpected error during prune_state: {e}", exc_info=True)
            return past_key_values, attention_mask # Return original on other errors

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