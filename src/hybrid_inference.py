"""
Hybrid Inference for Transformer-based LLMs on Mobile Devices

This module demonstrates a multi-pronged approach to efficient LLM inference
on mobile devices, as discussed in research on efficient inference techniques.
It combines four key optimization techniques:

1. Edge-Cloud Collaborative Inference: Dynamic partitioning of model layers
   between device and cloud based on network conditions and computational costs.

2. Runtime Token Pruning: Dynamically removing less important tokens during
   inference to reduce computational costs of attention operations.

3. Layer Compression and Skipping: Selectively compressing or skipping certain
   transformer layers based on their importance for the current inference task.

4. Model-Aware Memory Management: Efficient management of key-value cache using
   paging, quantization, and eviction strategies.

Together, these approaches enable efficient inference of large language models
on resource-constrained mobile devices.
"""

import logging
import numpy as np
import random
import torch
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import torch.nn.functional as F
import threading
import functools

# Import the four key modules
from edge_cloud.edge_cloud_manager import EdgeCloudManager
from token_pruning.token_pruner import TokenPruner
from layer_compression.layer_compression_skipping import LayerCompressionAndSkipping
from memory_manager.memory_manager import MemoryManager

# <<< ADDED: Import transformers >>>
from transformers import AutoModelForCausalLM, AutoTokenizer

# <<< ADDED: Import PEFT >>>
from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
# logging.basicConfig(level=logging.INFO)
# <<< Adjusted Logging >>>
log_level = logging.DEBUG if 'args' in locals() and args.detailed else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hybrid_inference")
# --------------------

# <<< Global skip counter and hook handles >>>
# Moved stats dictionary inside run_inference
hook_handles = []

# <<< Define the Pre-Forward Hook for Layer Skipping >>>
def layer_skip_pre_hook(module, args, layer_idx, layer_handler, stats, detailed_logging):
    """
    Pre-forward hook to decide if a layer should be skipped.
    Sets a flag on the module instance if skipping is decided.
    """
    # global hook_handles # Not strictly needed here, but good practice if modifying
    
    # The primary input to the layer is usually the first argument (hidden_states)
    hidden_state = args[0] if args else None
    module._skip_this_forward = False # Default: don't skip

    if layer_handler and hidden_state is not None:
        # Check if not a hot path layer
        if layer_idx not in layer_handler.hot_path_indices:
            try:
                should_skip_decision = layer_handler.should_skip_layer(layer_idx, hidden_state)
                if should_skip_decision:
                    module._skip_this_forward = True
                    stats["layers_skipped_decision_count"] += 1 # Increment counter here
                    if detailed_logging:
                        # Use logger.debug for potentially verbose info
                        logger.debug(f"--- Hook decision: Skip Layer {layer_idx} ---")
                # Optional: Log execution decision too
                # elif detailed_logging:
                #      logger.debug(f"--- Hook decision: Execute Layer {layer_idx} ---")

            except Exception as skip_e:
                logger.error(f"Error in layer_skip_pre_hook for layer {layer_idx}: {skip_e}", exc_info=True) # Added exc_info
                module._skip_this_forward = False # Default to not skipping on error
    
    # This hook doesn't modify the *input* args, just sets a flag on the module
    return None # Pre-hooks should return None or a tuple of modified args

def run_inference(
    model_name: str, 
    input_tokens: torch.Tensor, 
    prompt: str,
    max_new_tokens: int = 20, 
    enable_token_pruning: bool = True,
    enable_layer_optimization: bool = True,
    enable_memory_management: bool = True,
    enable_edge_cloud: bool = True, # Keep flag, logic commented
    max_memory_mb: int = 10,    # <<< Default low budget >>>
    detailed_logging: bool = False, # <<< Default False >>>
    device_map: str = "auto"       # <<< Default auto >>>
) -> Tuple[str, Dict[str, Any]]: # <<< Corrected return type hint >>>
    """
    Run hybrid inference using a real LLM with manual layer loop.
    """
    # Start timing
    start_time_inference = time.time() # More specific name
    step_times = [] # Still useful for avg token time
    
    # Determine device
    if device_map == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_map)
    
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}...")
    try:
        # Use torch.float16 for potentially smaller memory footprint if GPU available
        # Use torch.bfloat16 if available and preferred (Ampere+ GPUs)
        dtype = torch.float32 # Default
        if device.type == 'cuda':
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16 
                logger.info("Using bfloat16 for model loading on CUDA.")
            else:
                dtype = torch.float16
                logger.info("Using float16 for model loading on CUDA.")
        else:
            logger.info("Using float32 for model loading on CPU.")
            
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True) # Added exc_info
        raise
        
    model_config = model.config # Get config from loaded model
    
    # Extract model dimensions (ensure correct attribute names)
    hidden_size = getattr(model_config, "hidden_size", getattr(model_config, "n_embd", 768))
    num_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", 12))
    num_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "n_head", 12))
    vocab_size = getattr(model_config, "vocab_size", 50257) # Default for GPT-2
    
    # Ensure input tokens are on the correct device
    input_tokens = input_tokens.to(device)
    
    logger.info("=====================================================")
    logger.info("HYBRID INFERENCE - REAL MODEL EXECUTION")
    logger.info("=====================================================")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Model dimensions: {hidden_size}d, {num_layers} layers, {num_heads} attention heads")
    logger.info(f"Prompt: \"{prompt}\"")
    logger.info(f"Max New Tokens: {max_new_tokens}")
    logger.info(f"Enabled optimizations:")
    logger.info(f"  - Token Pruning: {enable_token_pruning}")
    logger.info(f"  - Layer Opt (Skip/Compress): {enable_layer_optimization}")
    logger.info(f"  - Memory Management (Quant): {enable_memory_management}")
    logger.info(f"  - Edge-Cloud Partitioning: {enable_edge_cloud} (Logic not active)")
    logger.info("=====================================================")
    
    # Initialize Managers
    memory_manager = None
    if enable_memory_management:
        logger.info("Initializing Memory Manager...")
        try:
            memory_manager = MemoryManager(
                max_memory_mb=max_memory_mb,
                quantization_enabled=True, # If flag is true, enable quantization
                enable_logging=detailed_logging
            )
            memory_manager.initialize_kv_cache(model_config) # <<< CORRECTED method name >>>
        except Exception as mem_init_e:
            logger.error(f"Failed to initialize MemoryManager: {mem_init_e}", exc_info=True)
            memory_manager = None # Ensure it's None on error

    token_pruner = None
    if enable_token_pruning:
        logger.info("Initializing Token Pruner...")
        try:
            token_pruner = TokenPruner(
                pruning_threshold=0.01, # <<< Drastically LOWERED threshold >>>
            )
        except Exception as pruner_init_e:
            logger.error(f"Failed to initialize TokenPruner: {pruner_init_e}", exc_info=True)
            token_pruner = None

    layer_handler = None
    if enable_layer_optimization:
        logger.info("Initializing Layer Compression/Skipping Manager...")
        # Define hot path layers (e.g., first, last, maybe middle)
        hot_path_layers = [0, 1, num_layers - 1] if num_layers > 2 else list(range(num_layers)) # Example
        try:
            layer_handler = LayerCompressionAndSkipping(
                model=model,
                compression_rank=-1, # Not used for skipping currently
                hot_path_indices=hot_path_layers,
                skip_threshold=0.5 # <<< INCREASED threshold to encourage skipping >>>
            )
            # Apply offline compression if desired (can be separated later)
            # logger.info("Applying offline layer compression (SVD)...")
            # layer_handler.apply_low_rank_factorization() # <-- KEEP COMMENTED
            # logger.info("Offline compression finished.")
        except Exception as layer_init_e:
            logger.error(f"Failed to initialize LayerCompressionAndSkipping: {layer_init_e}", exc_info=True)
            layer_handler = None # Ensure it's None on error

    # edge_cloud_manager = EdgeCloudManager(...) # Keep commented
    
    # <<< ADDED: Initialize EdgeCloudManager >>>
    edge_manager = None
    if enable_edge_cloud:
        logger.info("Initializing Edge-Cloud Manager...")
        try:
            # Import necessary class
            from edge_cloud.edge_cloud_manager import EdgeCloudManager 
            # <<< FIXED: Pass arguments expected by EdgeCloudManager __init__ >>>
            edge_manager = EdgeCloudManager(model=model) # Pass model, add others later if needed
            # Provide necessary config like num_layers
        except Exception as edge_init_e:
            logger.error(f"Failed to initialize EdgeCloudManager: {edge_init_e}", exc_info=True)
            edge_manager = None
    # -------------------------------------
    
    # Initialize stats dictionary
    stats = {
        "total_tokens_processed": input_tokens.shape[1],
        "generated_tokens": 0,
        "layers_skipped_decision_count": 0, # Counter incremented by hook
        "tokens_pruned_count": 0, # <<< ADDED for pruning >>>
        "step_details": []
    }
    global hook_handles # Declare we might modify the global list
    hook_handles = [] # Clear any previous handles
    
    # --- Find Model Layers and Embeddings --- (Consolidated)
    layers = None
    embedding_layer = None
    pos_embedding_layer = None
    final_norm = None
    lm_head = None
    is_gpt2_style = hasattr(model, 'transformer') and hasattr(model.transformer, 'h')
    is_opt_style = hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers')

    try: # Wrap identification in try/except
        if is_gpt2_style:
            logger.debug("Identified GPT-2 style model structure.")
            layers = model.transformer.h
            embedding_layer = model.transformer.wte
            pos_embedding_layer = model.transformer.wpe
            final_norm = model.transformer.ln_f
            lm_head = model.lm_head
        elif is_opt_style:
            logger.debug("Identified OPT style model structure.")
            layers = model.model.decoder.layers
            embedding_layer = model.model.decoder.embed_tokens
            pos_embedding_layer = model.model.decoder.embed_positions
            final_norm = model.model.decoder.final_layer_norm
            lm_head = model.lm_head
        else:
             # If not known structure, manual loop cannot proceed
             logger.error("Model architecture not recognized for manual loop.")
             # Return empty results or raise error
             return "", stats # Or raise NotImplementedError
             
        # Verify all components were found
        if not all([layers is not None, embedding_layer is not None, 
                   pos_embedding_layer is not None, final_norm is not None, 
                   lm_head is not None]):
            raise ValueError("Could not identify all required model components (layers, embeddings, norm, head).")
            
    except Exception as e:
        logger.error(f"Error identifying model components: {e}", exc_info=True)
        return "", stats # Return empty results on error

    # --- Register Hooks (if layer optimization is enabled) ---
    if enable_layer_optimization and layer_handler and layers:
        logger.info(f"Registering pre-forward hooks for layer skipping on {len(layers)} layers...")
        for i, layer_module in enumerate(layers):
            # Use functools.partial to pass extra arguments (layer_idx, handler, stats) to the hook
            partial_hook = functools.partial(
                layer_skip_pre_hook, 
                layer_idx=i, 
                layer_handler=layer_handler,
                stats=stats, # Pass stats dict to the hook
                detailed_logging=detailed_logging 
            )
            try:
                handle = layer_module.register_forward_pre_hook(partial_hook)
                hook_handles.append(handle)
            except Exception as hook_reg_e:
                logger.error(f"Failed to register hook for layer {i}: {hook_reg_e}")
        logger.info(f"Successfully registered {len(hook_handles)} hooks.")
    # --------------------------------------------------------

    # <<< Preparation for Manual Loop >>>
    current_tokens = input_tokens
    generated_token_ids = []
    past_key_values = None
    # Attention mask starts with shape [batch_size, input_len]
    attention_mask = torch.ones(current_tokens.shape, dtype=torch.long, device=device)

    logger.info(f"Starting inference loop for {max_new_tokens} tokens...")
    start_loop_time = time.time() # Time just the loop

    # Initialization for tracking
    token_pruning_actions = 0
    layer_skipping_decisions = 0
    total_generated_tokens = 0
    total_offloaded_layers_count = 0
    total_pruned_tokens_count = 0 # <<< Initialize prune counter >>>
    start_time = time.time()

    # <<< Manual layer-by-layer generation loop >>>
    try: # Wrap the main loop in try/except
        with torch.no_grad():
            # Configure output flags
            output_attentions_flag = enable_token_pruning and token_pruner # <<< Enable only if needed >>>
            output_hidden_states_flag = False
            # ---------------------------

            for i in range(max_new_tokens):
                step_start_time = time.time()
                current_seq_len_before_step = attention_mask.shape[1]
                if detailed_logging:
                    logger.debug(f"\n--- Gen Step {i+1}/{max_new_tokens} --- Seq Len: {current_seq_len_before_step} ---")

                # <<< ADDED: Edge-Cloud Partition Decision >>>
                local_layers = list(range(num_layers)) # Default: all local
                remote_layers = []
                if enable_edge_cloud and edge_manager:
                    try:
                        # Example: Pass minimal state for now
                        partition_decision = edge_manager.decide_partition(current_step=i)
                        local_layers = partition_decision.get("local_layers", local_layers)
                        remote_layers = partition_decision.get("remote_layers", remote_layers)
                        # <<< ADDED: Log the actual lists being used >>>
                        logger.info(f"  Step {i+1} Partition: Local={local_layers}, Remote={remote_layers}")
                        # -----------------------------------------
                    except Exception as part_e:
                        logger.error(f"Error getting partition decision: {part_e}", exc_info=True)
                # -----------------------------------------

                # --- Prepare inputs for this step --- 
                inputs_embeds = embedding_layer(current_tokens)
                
                # Calculate position IDs for the *new* token(s)
                past_kv_len = past_key_values[0][0].shape[-2] if past_key_values else 0 # Key dim is length
                current_input_len = current_tokens.shape[1] # Usually 1 after first step
                position_ids = torch.arange(past_kv_len, past_kv_len + current_input_len, dtype=torch.long, device=device).unsqueeze(0)
                
                position_embeds = pos_embedding_layer(position_ids)
                hidden_state = inputs_embeds + position_embeds
                # -----------------------------------
                
                # --- Layer Loop --- 
                new_past_key_values_list = [] # Store KV states for the *next* step
                all_layer_attentions = [] if output_attentions_flag else None # <<< Collect attentions >>>
                
                for layer_idx, layer in enumerate(layers):
                    layer_past = past_key_values[layer_idx] if past_key_values else None
                    layer_input_hidden_state = hidden_state
                    present_key_value = None # Initialize for the current layer

                    # --- Decide Local vs Remote Execution --- 
                    if layer_idx in local_layers:
                        # --- Execute Layer Locally --- 
                        if detailed_logging:
                             logger.debug(f"    Executing Layer {layer_idx} Locally...")
                             
                        # Per-Layer Dequantize Call (if needed)
                        if enable_memory_management and memory_manager:
                            memory_manager._dequantize_layer(layer_idx) 
                        
                        # Check skip flag set by hook (if hooks are active)
                        execute_layer_normally = True 
                        if enable_layer_optimization and hasattr(layer, '_skip_this_forward') and layer._skip_this_forward:
                            execute_layer_normally = False
                            layer._skip_this_forward = False # Reset flag
                            
                            # Handle Skip (within local execution)
                            hidden_state = layer_input_hidden_state # Pass hidden state through
                            present_key_value = layer_past # Preserve the past KV state
                            # Logging is done in hook

                        if execute_layer_normally:
                            # Execute Layer Normally
                            layer_outputs = layer(
                                layer_input_hidden_state,
                                layer_past=layer_past,
                                attention_mask=None, # Pass None, let layer handle causal mask
                                use_cache=True,
                                output_attentions=output_attentions_flag, 
                            )
                            
                            # Extract outputs
                            layer_attentions = None 
                            if isinstance(layer_outputs, tuple):
                                hidden_state = layer_outputs[0]
                                present_key_value = layer_outputs[1] if len(layer_outputs) > 1 else None
                                if output_attentions_flag and len(layer_outputs) > 2:
                                    layer_attentions = layer_outputs[2]
                            else: # Handle BaseMo...Output objects
                                hidden_state = layer_outputs.last_hidden_state
                                present_key_value = layer_outputs.past_key_value
                                if output_attentions_flag:
                                    layer_attentions = layer_outputs.attentions
                                    
                        if output_attentions_flag and layer_attentions is not None:
                             all_layer_attentions.append(layer_attentions) # Store attention for locally executed layers
                             
                    elif layer_idx in remote_layers:
                        # --- Simulate Remote Execution Placeholder --- 
                        if detailed_logging:
                             logger.debug(f"    Simulating Remote Execution for Layer {layer_idx}...")
                        
                        # <<< ADDED: Explicit log *inside* the remote block >>>
                        logger.info(f"    [Offload Check] Layer {layer_idx} is in remote_layers: {remote_layers}. Incrementing counter.")
                        # -------------------------------------------------
                        total_offloaded_layers_count += 1 # <<< Increment counter >>>
                        
                        # Placeholder: Assume instantaneous, perfect execution
                        # Hidden state passes through unchanged for now
                        hidden_state = layer_input_hidden_state 
                        
                        # Carry forward the previous KV state
                        present_key_value = layer_past 
                        
                        # TODO: Implement actual remote call / simulation
                        # 1. Serialize layer_input_hidden_state & maybe layer_past?
                        # 2. Send data (simulate network delay)
                        # 3. Receive results (simulate network delay + cloud compute delay)
                        # 4. Deserialize results into hidden_state and present_key_value
                        # --------------------------------------------
                    else:
                        # This case should not happen if partition covers all layers
                        logger.warning(f"Layer {layer_idx} not found in local or remote lists. Skipping.")
                        hidden_state = layer_input_hidden_state
                        present_key_value = layer_past

                    # <<< ADDED: Log shape of KV state being carried forward >>>
                    if detailed_logging:
                         kv_shape_log = (present_key_value[0].shape, present_key_value[1].shape) if present_key_value else None
                         run_location = "Local" if layer_idx in local_layers else "Remote (Sim)" if layer_idx in remote_layers else "ERROR"
                         skipped_flag = (not execute_layer_normally) if layer_idx in local_layers else False # Only log skip if local
                         logger.debug(f"    Layer {layer_idx} OUT: Carry KV Shape: {kv_shape_log} (Location: {run_location}, Skipped: {skipped_flag})")
                    # --------------------------------------------------------
                    new_past_key_values_list.append(present_key_value)
                
                # <<< RESTORED: Update past_key_values for the next iteration >>>
                past_key_values = tuple(new_past_key_values_list)
                # --------------------------------------------------------------

                # --- Final Processing After All Layers --- 
                hidden_state_before_norm = hidden_state # Keep for potential pruning scoring
                hidden_state = final_norm(hidden_state_before_norm)
                logits = lm_head(hidden_state)
                # -------------------------------------------
                
                # --- Token Pruning Hook --- 
                if token_pruner:
                    try:
                        # 1. Score Tokens (Requires Attention Output)
                        # Ensure output_attentions_flag was True during layer loop
                        if not output_attentions_flag or not all_layer_attentions:
                            logger.warning("Token pruning requires attention scores, but they were not collected.")
                            prune_indices = []
                        else:
                            last_layer_attentions = all_layer_attentions[-1] # Assuming we need the last layer's attention
                            # Determine the context length for scoring (usually KV cache length)
                            kv_len_for_scoring = past_key_values[0][0].shape[-2] if past_key_values and past_key_values[0] and past_key_values[0][0] is not None else 0
                            current_token_indices = list(range(kv_len_for_scoring))
                            
                            if detailed_logging:
                                logger.debug(f"  Scoring {len(current_token_indices)} tokens for pruning using last layer attention shape {last_layer_attentions.shape}")
                            
                            token_scores = token_pruner.score_tokens(
                                attention_scores=last_layer_attentions,
                                token_indices=current_token_indices # Pass indices corresponding to attention scores
                            )
                            
                            # 2. Identify Prunable Tokens
                            if token_scores is not None:
                                prune_indices = token_pruner.identify_prunable_tokens()
                            else:
                                prune_indices = []
                                if detailed_logging:
                                    logger.debug("  Token scoring returned None.")
                        
                        # 3. Apply Pruning if needed
                        if prune_indices:
                            # <<< ADDED: Check if past_key_values is None before pruning >>>
                            if past_key_values is None:
                                if detailed_logging:
                                    logger.debug(f"  Skipping pruning at step {i+1}: past_key_values is None.")
                            # -----------------------------------------------------------
                            elif detailed_logging:
                                logger.debug(f"  Token Pruning triggered at step {i+1}. Indices to prune: {prune_indices}")
                            
                            original_len = attention_mask.shape[-1]
                            # Calculate keep_indices (indices NOT in prune_indices)
                            all_indices = set(range(original_len))
                            prune_indices_set = set(prune_indices)
                            keep_indices = sorted(list(all_indices - prune_indices_set))
                            
                            # Ensure keep_indices are valid before proceeding
                            if not keep_indices or max(keep_indices) >= original_len:
                                logger.error(f"Invalid keep_indices calculated: {keep_indices} for original_len {original_len}. Skipping prune.")
                            else:
                                # Call prune_state with attention_mask, past_key_values, and token_indices_to_prune
                                # <<< FIXED: Unpack the tuple returned by prune_state >>>
                                pruned_past_key_values, pruned_attention_mask = token_pruner.prune_state(
                                    attention_mask=attention_mask,
                                    past_key_values=past_key_values,
                                    token_indices_to_prune=prune_indices # Pass indices to remove
                                )
                                
                                # Update attention_mask and past_key_values from the returned tuple
                                attention_mask = pruned_attention_mask
                                past_key_values = pruned_past_key_values
                                
                                pruned_count = original_len - attention_mask.shape[-1]
                                total_pruned_tokens_count += pruned_count
                                if detailed_logging:
                                    logger.debug(f"    Pruned {pruned_count} tokens. New length: {attention_mask.shape[-1]}")
                                
                                # <<< FIXED: Use the *original* hidden state for final layers >>>
                                # Note: After pruning, hidden_state needs to go through final_norm and lm_head again!
                                hidden_state = final_norm(hidden_state_before_norm) # Apply final norm to original state
                                logits = lm_head(hidden_state) # Recalculate logits
                                
                        # else: # No indices to prune
                        #     if detailed_logging:
                        #         logger.debug(f"  Token Pruning decision at step {i+1}: Keep all.")
                        #     # <<< If no pruning, still need to calculate logits >>>
                        #     hidden_state = final_norm(hidden_state_before_norm)
                        #     logits = lm_head(hidden_state)
                            
                    except Exception as prune_e:
                        logger.error(f"Error during token pruning: {prune_e}", exc_info=True)
                        token_pruner = None # Disable on error

                # --- Token Selection --- (Uses potentially recalculated logits)
                next_token_logits = logits[:, -1, :] # Shape: [batch_size, vocab_size]
                
                # Sampling parameters
                temperature = 0.7 
                top_k = 50       
                no_repeat_ngram_size = 2 # <<< ADDED >>>
                
                if temperature > 0:
                    # Apply temperature
                    scaled_logits = next_token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Apply Top-K filtering
                    if top_k > 0:
                        top_k = min(top_k, probs.size(-1)) 
                        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                        filter_mask = torch.zeros_like(probs)
                        filter_mask.scatter_(dim=-1, index=top_k_indices, src=torch.ones_like(top_k_probs))
                        filtered_probs = probs * filter_mask
                        # Use filtered_probs for further processing
                        sampling_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
                    else:
                        # If no top_k, use original probs
                        sampling_probs = probs

                    # <<< ADDED: No Repeat N-Gram Logic >>>
                    if no_repeat_ngram_size > 0 and len(generated_token_ids) >= no_repeat_ngram_size - 1:
                        # Get the n-1 tokens preceding the current position
                        # Assuming batch size is 1 for generation
                        prefix_tokens = generated_token_ids[-(no_repeat_ngram_size - 1):]
                        # Find potential banned tokens
                        banned_tokens = []
                        # Iterate through possible next tokens (can optimize this later)
                        # For now, consider all tokens. Could restrict to top_k indices?
                        for token_id in range(vocab_size):
                            potential_ngram = prefix_tokens + [token_id]
                            # Check if this ngram has occurred before in the sequence
                            # Convert generated_token_ids to a list of ngrams
                            # This check is inefficient for long sequences, should be optimized if needed
                            for j in range(len(generated_token_ids) - no_repeat_ngram_size + 1):
                                existing_ngram = generated_token_ids[j : j + no_repeat_ngram_size]
                                if potential_ngram == existing_ngram:
                                    banned_tokens.append(token_id)
                                    break # No need to check further for this token_id
                                    
                        if banned_tokens:
                             # Set probability of banned tokens to 0
                             # Ensure banned_tokens are valid indices
                             banned_indices = torch.tensor(banned_tokens, dtype=torch.long, device=device)
                             # Clamp indices to be within vocab size just in case
                             banned_indices = torch.clamp(banned_indices, 0, vocab_size - 1)
                             sampling_probs.scatter_(dim=-1, index=banned_indices.unsqueeze(0), value=0.0) # Use unsqueeze for batch dim
                             
                             # Renormalize probabilities after banning tokens
                             renorm_sum = torch.sum(sampling_probs, dim=-1, keepdim=True)
                             if renorm_sum > 1e-9: # Avoid division by zero if all tokens banned
                                 sampling_probs = sampling_probs / renorm_sum
                             else:
                                 # If all likely tokens were banned, maybe fall back or raise warning?
                                 logger.warning("All probable tokens banned by no_repeat_ngram_size. Sampling might be random.")
                                 # Fallback: uniform distribution over all tokens? Or keep original filtered probs?
                                 # For now, let it sample from the zero distribution (will likely pick 0)
                                 pass 
                    # -------------------------------------

                    # Sample from the (potentially filtered and renormalized) distribution
                    next_token_id = torch.multinomial(sampling_probs, num_samples=1)
                    
                else:
                    # Fallback to argmax if temperature is 0
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                # -----------------------
                
                # --- Update state for next iteration --- 
                generated_token_ids.append(next_token_id.item()) # Store ID
                current_tokens = next_token_id # Next input is the just generated token
                # Append a mask token for the token just generated
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device)], dim=1)
                # -------------------------------------

                # Call Memory Manager Update (after state is updated)
                if enable_memory_management and memory_manager:
                    # <<< ADDED: Call update_state to track real KV cache >>>
                    # Pass the loop's *current* past_key_values, which might have been pruned (but pruning is disabled now)
                    memory_manager.update_state(past_key_values, attention_mask.shape[1])
                    # ---------------------------------------------------------
                    memory_manager.check_memory_usage_and_trigger_actions()
                
                # Record step timing & stats
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                step_times.append(step_duration)
                stats["generated_tokens"] += 1
                stats["total_tokens_processed"] += 1 # Increment total processed count
                current_mem_usage = 0.0
                if memory_manager:
                    # Fetch potentially updated stats after actions
                    current_mem_usage = memory_manager.get_stats().get("current_memory_usage_mb", 0.0)
                stats["step_details"].append({
                    "step": i + 1,
                    "token_id": next_token_id.item(),
                    "token_text": tokenizer.decode(next_token_id.squeeze()),
                    "time_ms": step_duration * 1000,
                    "memory_usage_mb": current_mem_usage, 
                })
                
                if detailed_logging:
                    token_text = tokenizer.decode(next_token_id.squeeze())
                    logger.debug(f"  Generated token {i+1}: ID={next_token_id.item()}, Text=\"{token_text}\" | Time: {step_duration:.4f}s")
                    if memory_manager:
                        logger.debug(f"    Memory after step: {current_mem_usage:.4f} MB")
                         
                # Check for EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    logger.info(f"EOS token generated. Stopping inference at step {i+1}.")
                    break
                    
    except Exception as loop_e:
        logger.error(f"Error occurred during generation loop: {loop_e}", exc_info=True)
        # Loop might terminate early, results may be partial
        
    finally:
        # <<< ADDED: Remove hooks after generation (in finally block) >>>
        if hook_handles:
            logger.info(f"Removing {len(hook_handles)} hooks...")
            for handle in hook_handles:
                try:
                    handle.remove()
                except Exception as hook_remove_e:
                     logger.warning(f"Error removing hook: {hook_remove_e}")
            hook_handles = [] # Clear the list
            logger.info("Hooks removed.")
        # -------------------------------------------------------------

    end_loop_time = time.time()
    total_loop_time = end_loop_time - start_loop_time
    total_inference_time = end_loop_time - start_time_inference

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(input_tokens[0].tolist() + generated_token_ids, skip_special_tokens=True)

    # Summarize results
    logger.info("")
    logger.info("="*53)
    logger.info("Inference Summary (Manual Loop with Hooks)")
    logger.info("="*53)
    logger.info(f"Total Inference Time: {total_inference_time:.2f}s")
    logger.info(f"Generation Loop Time: {total_loop_time:.2f}s")
    if stats['generated_tokens'] > 0 and len(step_times) > 0:
        avg_token_time = sum(step_times) / len(step_times)
        logger.info(f"Average time per generated token: {avg_token_time:.4f}s ({1/avg_token_time:.2f} tokens/s)")
    logger.info(f"Generated {stats['generated_tokens']} new tokens")
    logger.info(f"Layers skipped (decision count): {stats['layers_skipped_decision_count']}")
    logger.info(f"Tokens Pruned: {total_pruned_tokens_count}")
    logger.info(f"Layers Offloaded: {total_offloaded_layers_count}")
    logger.info(f"Final sequence length: {attention_mask.shape[1]}")

    if memory_manager:
        logger.info("--- Memory Manager Stats ---")
        try:
            mem_stats = memory_manager.get_stats()
            # <<< ADDED: Log the received dictionary >>>
            logger.debug(f"Stats dict received by hybrid_inference summary block: {mem_stats}")
            # ---------------------------------------
            
            # <<< ADDED: Debug before Peak Memory log >>>
            peak_mem_val = mem_stats.get('peak_memory_usage_mb', 0.0)
            logger.debug(f"Value for Peak Memory Usage: {peak_mem_val} (Type: {type(peak_mem_val)})")
            # -----------------------------------------
            logger.info(f"Peak Memory Usage: {peak_mem_val:.2f} MB")
            
            # <<< ADDED: Debug before Quant Events log >>>
            quant_events_val = mem_stats.get('quantization_events', 0)
            logger.debug(f"Value for Quantization Events: {quant_events_val} (Type: {type(quant_events_val)})")
            # -------------------------------------------
            logger.info(f"Quantization Events: {quant_events_val}")
            
            # <<< ADDED: Debug before Quant Layers log >>>
            quant_layers_val = mem_stats.get('quantized_layers', set())
            quant_layers_list = sorted(list(quant_layers_val))
            logger.debug(f"Value for Quantized Layers: {quant_layers_list} (Type: {type(quant_layers_val)} -> {type(quant_layers_list)})")
            # -------------------------------------------
            logger.info(f"Layers Quantized ({len(quant_layers_list)}): {quant_layers_list}")
            
            # <<< ADDED: Debug before Dequant Events log >>>
            dequant_events_val = mem_stats.get('dequantization_events', 0)
            logger.debug(f"Value for Dequantization Events: {dequant_events_val} (Type: {type(dequant_events_val)})")
            # --------------------------------------------
            logger.info(f"Dequantization Events: {dequant_events_val}")
        except Exception as stats_e:
             logger.error(f"Error retrieving/displaying memory stats: {stats_e}")

    if layer_handler and enable_layer_optimization:
        logger.info("--- Layer Optimization Stats ---")
        try:
            layer_stats = layer_handler.get_metrics()
            # Use .get() for safety
            logger.info(f"Compression Ratio (Handler): {layer_stats.get('compression_ratio', 'N/A')}")
            logger.info(f"Total Layers Evaluated for Skipping (Handler): {layer_stats.get('total_layers_evaluated', 0)}")
            logger.info(f"Total Layers Skipped (Handler): {layer_stats.get('layers_skipped', 0)}") 
            logger.info(f"Skipping Efficiency (Handler): {layer_stats.get('skipping_efficiency', 0.0):.4f}")
        except Exception as stats_e:
             logger.error(f"Error retrieving/displaying layer stats: {stats_e}")

    # --- Final Generated Text --- 
    logger.info("--- Generated Text ---")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Final Output: {full_text}") # Keep the full output log
    # <<< FIXED: Replace newlines with spaces before logging >>>
    single_line_generated_text = generated_text.replace('\n', ' ').strip()
    logger.info(f"Newly Generated Text: {single_line_generated_text}") 

    logger.info("="*53)
    logger.info("Hybrid inference script finished.")

    return generated_text, stats

# === Main Execution Block ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Hybrid LLM Inference")
    
    # Model and Generation Parameters
    parser.add_argument("--model-name", type=str, default="gpt2", help="Name of the Hugging Face model to use (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Input prompt for the model")
    parser.add_argument("--tokens", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (e.g., cpu, cuda, cuda:0, auto)")

    # Optimization Flags (Defaults changed to False to enable step-by-step activation)
    parser.add_argument("--token-pruning", action="store_true", help="Enable runtime token pruning")
    parser.add_argument("--layer-opt", action="store_true", help="Enable layer compression/skipping")
    parser.add_argument("--memory-opt", action="store_true", help="Enable memory management (quantization)")
    parser.add_argument("--edge-cloud", action="store_true", help="Enable edge-cloud partition decision logic (no execution)")
    
    # Memory Manager Specific
    parser.add_argument("--mem-budget", type=int, default=100, help="Memory budget in MB for Memory Manager") # Increased default
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--detailed", action="store_true", help="Enable detailed DEBUG logging")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to save results and stats")
    
    args = parser.parse_args()
    
    # Set Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    # Adjust Log Level based on --detailed flag
    log_level = logging.DEBUG if args.detailed else logging.INFO
    # Update root logger level
    logging.getLogger().setLevel(log_level) 
    logger.setLevel(log_level) # Ensure our specific logger respects the level
    # Reconfigure basicConfig if needed (might be redundant if done earlier)
    # logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logger.info(f"Log level set to: {logging.getLevelName(logger.getEffectiveLevel())}")

    # --- Prepare Inputs --- 
    # Load tokenizer separately for input prep if run_inference raises early error
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # Add padding token if missing (like GPT-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for tokenizer.")
            
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    except Exception as tokenizer_e:
        logger.error(f"Failed to load tokenizer or encode prompt: {tokenizer_e}", exc_info=True)
        exit(1) # Exit if basic input prep fails
    # ---------------------

    # --- Run Inference --- 
    generated_text, stats = run_inference(
        model_name=args.model_name,
        input_tokens=input_ids,
        prompt=args.prompt, 
        max_new_tokens=args.tokens,
        enable_token_pruning=args.token_pruning,
        enable_layer_optimization=args.layer_opt,
        enable_memory_management=args.memory_opt,
        enable_edge_cloud=args.edge_cloud, 
        max_memory_mb=args.mem_budget,
        detailed_logging=args.detailed,
        device_map=args.device
    )
    # --------------------

    # --- Save Results (Optional) --- 
    if args.output:
        logger.info(f"Saving results to {args.output}...")
        results_data = {
            "args": vars(args),
            "generated_text": generated_text,
            "stats": stats
        }
        try:
            with open(args.output, 'w') as f:
                # Use helper function for potentially non-serializable items in stats
                def default_serializer(obj):
                    if isinstance(obj, set):
                        return sorted(list(obj))
                    if isinstance(obj, torch.Tensor):
                        return obj.tolist() # Or some representation
                    # Add other types as needed
                    try:
                         # Try standard JSON serialization first
                         return json.JSONEncoder().encode(obj)
                    except TypeError:
                         return str(obj) # Fallback to string representation
                
                json.dump(results_data, f, indent=2, default=default_serializer)
            logger.info("Results saved successfully.")
        except Exception as save_e:
            logger.error(f"Failed to save results to {args.output}: {save_e}", exc_info=True)
    # -----------------------------

    logger.info("Hybrid inference script finished.") 
