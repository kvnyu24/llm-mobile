"""
Mini-LLM Handler for Edge-Cloud Collaborative Inference

This module provides functionality to use a smaller local model for handling
"easy" tokens, which can be predicted without using the full model.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mini_llm")


class TokenDifficultyEstimator:
    """
    Estimates the difficulty of predicting each token in a sequence.
    
    This is used to decide whether to use a local mini-LLM or the full
    cloud model for each token in the sequence.
    """
    
    def __init__(self, difficulty_threshold: float = 0.7):
        """
        Initialize the token difficulty estimator.
        
        Args:
            difficulty_threshold: Threshold above which a token is considered "hard"
        """
        self.difficulty_threshold = difficulty_threshold
        self.token_stats = {}  # Track statistics about seen tokens
        
    def estimate_token_difficulty(self, 
                                  token_id: int, 
                                  context_ids: List[int],
                                  logits: Optional[np.ndarray] = None) -> float:
        """
        Estimate how difficult it is to predict a specific token.
        
        Args:
            token_id: The token ID to evaluate
            context_ids: Context tokens preceding this token
            logits: Probability distribution from model (if available)
            
        Returns:
            Difficulty score (0.0-1.0) where higher means more difficult
        """
        # Initialize token statistics if we haven't seen this token before
        if token_id not in self.token_stats:
            self.token_stats[token_id] = {
                "occurences": 0,
                "contexts": {},
                "avg_entropy": 0.0,
                "predictability": 0.5  # Start with neutral predictability
            }
            
        token_info = self.token_stats[token_id]
        token_info["occurences"] += 1
        
        # If we have logits, compute entropy-based difficulty
        if logits is not None:
            # Compute normalized probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            # Compute entropy: -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Update rolling average entropy
            token_info["avg_entropy"] = (token_info["avg_entropy"] * (token_info["occurences"] - 1) + entropy) / token_info["occurences"]
            
            # Get probability of the correct token
            correct_prob = probs[token_id]
            
            # Higher probability means lower difficulty
            score = 1.0 - correct_prob
            
            # Weight more heavily toward entropy for tokens we've seen more
            # Mix difficulty score with entropy-based score
            weight = min(0.8, token_info["occurences"] / 100)  # Max weight 0.8
            final_score = (1 - weight) * score + weight * (entropy / 5.0)  # Normalize entropy
            
            return final_score
            
        else:
            # Without logits, use context-based heuristics
            
            # Store context n-gram statistics
            # We use a 3-gram approach here
            context_len = min(3, len(context_ids))
            if context_len > 0:
                context_key = tuple(context_ids[-context_len:])
                
                if context_key not in token_info["contexts"]:
                    token_info["contexts"][context_key] = 1
                else:
                    token_info["contexts"][context_key] += 1
                
                # Compute what percentage of times this token follows this context
                total_context_occurences = sum(token_info["contexts"].values())
                context_probability = token_info["contexts"][context_key] / total_context_occurences
                
                # If token often follows this context, it's more predictable
                predictability = context_probability
                
                # Update rolling average predictability
                token_info["predictability"] = (token_info["predictability"] * (token_info["occurences"] - 1) + predictability) / token_info["occurences"]
            
            # Convert predictability to difficulty (inverse relationship)
            return 1.0 - token_info["predictability"]
            
    def is_token_hard(self, 
                      token_id: int, 
                      context_ids: List[int],
                      logits: Optional[np.ndarray] = None) -> bool:
        """
        Determine if a token is "hard" to predict and should use the full model.
        
        Args:
            token_id: The token ID to evaluate
            context_ids: Context tokens preceding this token
            logits: Probability distribution from model (if available)
            
        Returns:
            True if token is considered hard, False if easy
        """
        difficulty = self.estimate_token_difficulty(token_id, context_ids, logits)
        return difficulty > self.difficulty_threshold


class MiniLLMHandler:
    """
    Manages the use of a small local model for handling easy tokens.
    
    This implements the paper's approach for using a small local model for
    tokens that are easy to predict, while offloading hard tokens to the cloud.
    """
    
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer,
                 difficulty_estimator: Optional[TokenDifficultyEstimator] = None,
                 confidence_threshold: float = 0.8):
        """
        Initialize the Mini-LLM handler.
        
        Args:
            model: Small Transformer model for local inference
            tokenizer: Tokenizer for the model
            difficulty_estimator: Estimator for token difficulty
            confidence_threshold: Confidence threshold for accepting mini-LLM predictions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.difficulty_estimator = difficulty_estimator or TokenDifficultyEstimator()
        self.confidence_threshold = confidence_threshold
        self.stats = {
            "total_tokens": 0,
            "easy_tokens": 0,
            "hard_tokens": 0,
            "success_rate": 0.0
        }
        
    def predict_next_token(self, input_ids: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        Predict the next token using the mini-LLM.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Tuple of (predicted_token_id, confidence, raw_logits)
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get the most likely token and its probability
            top_prob, top_id = torch.max(probs, dim=-1)
            
            return top_id.item(), top_prob.item(), logits
            
    def should_use_mini_llm(self, 
                            input_ids: torch.Tensor, 
                            token_history: Optional[List[int]] = None) -> bool:
        """
        Determine if the mini-LLM should be used for the next token.
        
        This uses the token difficulty estimator to assess whether the next
        token is likely to be easy enough for the mini-LLM to handle.
        
        Args:
            input_ids: Current input token IDs
            token_history: History of previously generated tokens and their difficulty
            
        Returns:
            True if mini-LLM should be used, False if full model needed
        """
        # If we have no history, we use a simple heuristic
        if token_history is None or len(token_history) < 10:
            # At the beginning or when we have limited history,
            # use mini-LLM with 50% probability as a baseline
            return np.random.random() < 0.5
            
        # Get context tokens from input
        context_ids = input_ids.cpu().numpy().tolist()[0]
        
        # Check recent token history
        # If we've had several hard tokens in a row, likely the next one is hard too
        hard_count = sum(1 for t in token_history[-5:] if t["hard"])
        
        if hard_count >= 4:
            # If 4+ of the last 5 tokens were hard, likely we're in a hard section
            return False
            
        if hard_count <= 1:
            # If only 0-1 of the last 5 tokens were hard, likely we're in an easy section
            return True
            
        # For the middle cases, make a real prediction using the difficulty estimator
        # Use a simulated token ID (we don't know the actual next token yet)
        # The estimator will primarily look at context patterns
        simulated_next_id = token_history[-1]["token_id"]  # Just reuse the last token as a placeholder
        difficulty = self.difficulty_estimator.estimate_token_difficulty(simulated_next_id, context_ids)
        
        return difficulty < self.difficulty_estimator.difficulty_threshold
        
    def evaluate_mini_llm_performance(self, 
                                      correct_token: int, 
                                      predicted_token: int, 
                                      confidence: float) -> Dict[str, Any]:
        """
        Evaluate how well the mini-LLM performed on a prediction.
        
        Args:
            correct_token: The actual correct token
            predicted_token: Token predicted by mini-LLM
            confidence: Confidence score for the prediction
            
        Returns:
            Performance metrics
        """
        self.stats["total_tokens"] += 1
        
        # Update token counts
        was_correct = (correct_token == predicted_token)
        was_confident = (confidence >= self.confidence_threshold)
        
        metrics = {
            "correct": was_correct,
            "confidence": confidence,
            "high_confidence": was_confident
        }
        
        if was_correct:
            # Token was correctly predicted by the mini-LLM
            self.stats["easy_tokens"] += 1
            metrics["token_type"] = "easy"
        else:
            # Token was mispredict, might be a hard token
            self.stats["hard_tokens"] += 1
            metrics["token_type"] = "hard"
            
        # Update success rate
        self.stats["success_rate"] = self.stats["easy_tokens"] / self.stats["total_tokens"]
        
        return metrics
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics for the mini-LLM.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_tokens": self.stats["total_tokens"],
            "easy_tokens": self.stats["easy_tokens"],
            "hard_tokens": self.stats["hard_tokens"],
            "easy_percentage": (self.stats["easy_tokens"] / max(1, self.stats["total_tokens"])) * 100,
            "success_rate": self.stats["success_rate"]
        } 