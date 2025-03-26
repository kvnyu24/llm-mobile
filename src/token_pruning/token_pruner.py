"""
Runtime Token Pruning for LLM inference.

This module implements token pruning techniques to reduce computation cost
during autoregressive decoding in transformer-based language models.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
    
    def __init__(self, pruning_threshold=0.01, max_shadow_size=100):
        """
        Initialize the Token Pruner.
        
        Args:
            pruning_threshold: Threshold below which tokens are considered low-impact
            max_shadow_size: Maximum number of tokens to keep in the shadow set
        """
        self.pruning_threshold = pruning_threshold
        self.max_shadow_size = max_shadow_size
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
        
        As described in the paper, we derive importance scores Sⱼ from 
        attention weights αᵢⱼ for each token.
        
        Args:
            attention_scores: Attention matrices from the model [batch, heads, seq_len, seq_len]
            token_indices: Indices of the tokens to score
            
        Returns:
            Dictionary mapping token indices to importance scores
        """
        start_time = time.time()
        
        # If attention_scores is a tensor (e.g., PyTorch), convert to numpy
        if HAS_TORCH and isinstance(attention_scores, torch.Tensor):
            attention_scores = attention_scores.detach().cpu().numpy()
            
        # Get dimensions
        n_heads = attention_scores.shape[1]
        seq_len = attention_scores.shape[2]
        
        # Calculate importance score for each token (averaged across heads)
        # Using maximum attention received by each token as importance
        scores = {}
        
        # For each token position
        for idx in token_indices:
            if idx >= seq_len:
                continue
            
            # Get attention received by token idx from all other tokens across all heads
            # attention_scores has shape [batch, heads, seq_len, seq_len]
            # We take attention_scores[0, :, :, idx] to get attention TO token idx
            attn_to_token = attention_scores[0, :, :, idx]  # Shape: [heads, seq_len]
            
            # Calculate maximum attention score for each head
            max_attns = np.max(attn_to_token, axis=1)  # Shape: [heads]
            
            # Average across heads as the importance score
            score = np.mean(max_attns)
            scores[idx] = float(score)
        
        # Update token scores dictionary
        self.token_scores.update(scores)
        
        # Update metrics
        self.metrics["score_time"] += time.time() - start_time
        self.metrics["score_calls"] += 1
        
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
        protected_tokens = 3  
        
        for idx, score in self.token_scores.items():
            # Skip protected tokens at the beginning of the sequence
            if idx < protected_tokens:
                continue
            
            # If score is below threshold, mark for pruning
            if score < self.pruning_threshold:
                prunable_tokens.append(idx)
        
        return prunable_tokens
        
    def prune_tokens(self, tokens, hidden_states):
        """
        Remove low-impact tokens from the active sequence.
        
        As described in the paper, we physically remove pruned tokens from 
        computation and store them in a "shadow set" for potential reintroduction.
        
        Args:
            tokens: The current token sequence
            hidden_states: The model's hidden states for these tokens
            
        Returns:
            Pruned token sequence and updated hidden states
        """
        start_time = time.time()
        
        # Identify tokens to prune
        prunable_indices = self.identify_prunable_tokens()
        
        if not prunable_indices:
            return tokens, hidden_states
            
        # Convert to set for faster lookups
        prunable_set = set(prunable_indices)
        
        # Determine which tokens to keep
        keep_indices = [i for i in range(len(tokens)) if i not in prunable_set]
        
        # Save pruned tokens to shadow set
        self._add_to_shadow_set(tokens, hidden_states, prunable_indices)
        
        # Create pruned token list
        pruned_tokens = [tokens[i] for i in keep_indices]
        
        # Prune hidden states
        if hidden_states is not None:
            # If hidden_states is a tensor (e.g., PyTorch)
            if HAS_TORCH and isinstance(hidden_states, torch.Tensor):
                device = hidden_states.device
                keep_indices_tensor = torch.tensor(keep_indices, device=device)
                pruned_hidden_states = torch.index_select(hidden_states, 1, keep_indices_tensor)
            else:
                # Assume numpy array or similar
                pruned_hidden_states = np.take(hidden_states, keep_indices, axis=1)
        else:
            pruned_hidden_states = None
            
        # Update pruning statistics
        self.pruning_count += len(prunable_indices)
        self.metrics["tokens_pruned"] += len(prunable_indices)
        self.metrics["tokens_seen"] += len(tokens)
        self.metrics["pruning_time"] += time.time() - start_time
        self.metrics["pruning_calls"] += 1
        
        return pruned_tokens, pruned_hidden_states
        
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