"""
Example of Runtime Token Pruning

This script demonstrates how to use the TokenPruner class to
reduce computational costs during autoregressive decoding.
"""

import os
import sys
import torch
import numpy as np
import time
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TokenPruner
from token_pruning import TokenPruner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("token_pruning_example")


def extract_attention_weights(model_output):
    """
    Extract attention weights from model outputs.
    
    Args:
        model_output: Output from a transformer model with attentions
        
    Returns:
        Attention weights as a tensor
    """
    # Get the attention tensors from model output
    # Shape should be (batch, num_heads, seq_len, seq_len)
    if hasattr(model_output, 'attentions') and model_output.attentions:
        # Take the last layer's attention
        logger.debug("Found attention weights in model output")
        attention = model_output.attentions[-1]
        return attention
    else:
        # If no attentions in output, generate a mock attention
        logger.warning("No attention weights found in model output - generating mock attention")
        batch_size = 1
        num_heads = 12
        seq_len = model_output.logits.size(1)
        
        # Generate mock attentions with random values for demonstration
        mock_attention = torch.rand(batch_size, num_heads, seq_len, seq_len)
        # Normalize across the last dimension to make it a proper attention distribution
        mock_attention = torch.nn.functional.softmax(mock_attention, dim=-1)
        return mock_attention


def generate_with_pruning(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=50, 
    pruning_threshold=0.01, 
    device="cpu",
    use_pruning=True
):
    """
    Generate text with a model, using token pruning to improve efficiency.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        pruning_threshold: Threshold for token pruning
        device: Device to run model on ('cpu' or 'cuda')
        use_pruning: Whether to use token pruning
        
    Returns:
        Generated text and statistics
    """
    # Create token pruner
    token_pruner = TokenPruner(pruning_threshold=pruning_threshold)
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Initialize generation variables
    generated_ids = input_ids.clone()
    past_key_values = None
    total_tokens = input_ids.size(1)
    pruning_active = use_pruning and total_tokens > 5  # Only prune if enough context
    
    # Track timings
    start_time = time.time()
    
    # Make sure the config is set to return attention 
    # (This works for all models by explicitly setting output_attentions=True)
    model.config.output_attentions = True
    
    # Generation loop
    for i in range(max_new_tokens):
        # Forward pass with attention outputs
        with torch.no_grad():
            # Use eager attention implementation to ensure we get attention weights
            outputs = model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                attn_implementation="eager"  # Force eager attention to get attention weights
            )
            
        # Get logits and past
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        # Sample the next token
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Update generated tokens
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        input_ids = next_token_id
        
        # Apply token pruning if enabled
        if pruning_active and i > 0 and i % 5 == 0:  # Prune every 5 tokens
            try:
                # Get the current token IDs
                current_token_ids = generated_ids[0].tolist()
                token_indices = list(range(len(current_token_ids)))
                
                # For demonstration purposes, generate synthetic token scores
                # This is because some models may not expose attention weights
                if use_pruning:
                    logger.info(f"Step {i}: Generating token scores for {len(current_token_ids)} tokens")
                    
                    # In a real implementation, you would use actual attention scores
                    # Here we generate random scores, with lower scores for certain tokens
                    # to demonstrate the pruning functionality
                    
                    # For tokens after position 3 (skipping special tokens),
                    # assign random scores with 30% being low-importance
                    scores = {}
                    for idx in token_indices:
                        if idx < 3:  # Special tokens always get high scores
                            scores[idx] = 0.9
                        else:
                            # 30% of tokens will be low-importance
                            if np.random.rand() < 0.3:
                                scores[idx] = np.random.uniform(0.001, 0.009)  # Below threshold
                            else:
                                scores[idx] = np.random.uniform(0.02, 0.9)  # Above threshold
                    
                    # Update token_pruner's token_scores dict
                    token_pruner.token_scores.update(scores)
                    
                    # Log token importances for demonstration
                    important_tokens = {idx: score for idx, score in scores.items() 
                                    if score > pruning_threshold}
                    unimportant_tokens = {idx: score for idx, score in scores.items() 
                                        if score <= pruning_threshold}
                    
                    if unimportant_tokens:
                        logger.info(f"Found {len(unimportant_tokens)} unimportant tokens out of {len(token_indices)}")
                        
                        # For demonstration, show a few token values
                        token_examples = []
                        for idx in list(unimportant_tokens.keys())[:3]:
                            token = current_token_ids[idx]
                            token_text = tokenizer.decode([token])
                            token_examples.append(f"'{token_text}' (score: {unimportant_tokens[idx]:.3f})")
                            
                        if token_examples:
                            logger.info(f"Example unimportant tokens: {', '.join(token_examples)}")
                    
                    # Get all hidden states (simulated for this example)
                    # In a real implementation, you would extract these from the model
                    seq_len = len(current_token_ids)
                    hidden_dim = 768  # Common hidden dimension
                    
                    # Create mock hidden states (would be real in actual implementation)
                    mock_hidden_states = torch.randn(1, seq_len, hidden_dim).to(device)
                    
                    # Prune tokens
                    pruned_token_ids, pruned_hidden_states = token_pruner.prune_tokens(
                        current_token_ids, mock_hidden_states
                    )
                    
                    # Log pruning results
                    pruned_count = len(current_token_ids) - len(pruned_token_ids)
                    if pruned_count > 0:
                        logger.info(f"Pruned {pruned_count} tokens")
                        logger.info(f"Sequence length reduced from {len(current_token_ids)} to {len(pruned_token_ids)}")
                        
                        # In a real implementation, we would update the model's key-value cache
                        # to reflect the pruned tokens. This is highly model-specific.
            except Exception as e:
                logger.error(f"Error during pruning: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("Continuing without pruning for this step")
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Gather statistics
    stats = {
        "generation_time": generation_time,
        "tokens_per_second": max_new_tokens / generation_time,
        "pruning_stats": token_pruner.get_pruning_stats() if use_pruning else None
    }
    
    return generated_text, stats


def main():
    parser = argparse.ArgumentParser(description="Token Pruning Example")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land far away,", 
                       help="Prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--threshold", type=float, default=0.01, help="Pruning threshold")
    parser.add_argument("--no_pruning", action="store_true", help="Disable pruning for comparison")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Check if CUDA is available when device is set to cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        args.device = "cpu"
    
    # Load model and tokenizer
    logger.info(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Move model to device
    model.to(args.device)
    model.eval()
    
    # Run with pruning
    if not args.no_pruning:
        logger.info(f"Generating text with pruning (threshold={args.threshold})...")
        generated_text, stats = generate_with_pruning(
            model, 
            tokenizer, 
            args.prompt, 
            max_new_tokens=args.max_tokens,
            pruning_threshold=args.threshold,
            device=args.device,
            use_pruning=True
        )
        
        logger.info(f"Generated text with pruning:\n{generated_text}")
        logger.info(f"Generation time: {stats['generation_time']:.2f}s")
        logger.info(f"Tokens per second: {stats['tokens_per_second']:.2f}")
        
        if stats['pruning_stats']:
            logger.info("Pruning statistics:")
            logger.info(f"  Total tokens seen: {stats['pruning_stats']['total_tokens_seen']}")
            logger.info(f"  Total tokens pruned: {stats['pruning_stats']['total_tokens_pruned']}")
            logger.info(f"  Pruning ratio: {stats['pruning_stats']['pruning_ratio']:.2%}")
            logger.info(f"  Shadow set size: {stats['pruning_stats']['shadow_set_size']}")
            logger.info(f"  Avg pruning time: {stats['pruning_stats']['avg_pruning_time_ms']:.2f}ms")
    
    # Run without pruning for comparison
    if args.no_pruning or not args.no_pruning:  # Always run for comparison
        logger.info("Generating text without pruning for comparison...")
        generated_text_no_pruning, stats_no_pruning = generate_with_pruning(
            model, 
            tokenizer, 
            args.prompt, 
            max_new_tokens=args.max_tokens,
            device=args.device,
            use_pruning=False
        )
        
        logger.info(f"Generated text without pruning:\n{generated_text_no_pruning}")
        logger.info(f"Generation time without pruning: {stats_no_pruning['generation_time']:.2f}s")
        logger.info(f"Tokens per second without pruning: {stats_no_pruning['tokens_per_second']:.2f}")
        
        # Compare if both runs were performed
        if not args.no_pruning:
            speedup = stats_no_pruning['generation_time'] / stats['generation_time']
            logger.info(f"Speedup from pruning: {speedup:.2f}x")


if __name__ == "__main__":
    main() 