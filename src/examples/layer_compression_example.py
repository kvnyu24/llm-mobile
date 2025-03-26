#!/usr/bin/env python
"""
Layer Compression and Skipping Example

This example demonstrates how to use the LayerCompressionAndSkipping class
to reduce computational cost by applying low-rank factorization to model weights
and dynamically skipping layers during inference.
"""

import os
import sys
import time
import numpy as np
import argparse

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layer_compression.layer_compression_skipping import LayerCompressionAndSkipping

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    print("PyTorch or transformers not available. Using mock implementations.")
    HAS_TORCH = False

def create_mock_model():
    """Create a mock model for demonstration when PyTorch is not available."""
    class MockConfig:
        def __init__(self):
            self.num_hidden_layers = 12
            self.hidden_size = 768
    
    class MockLayer:
        def __init__(self, idx):
            self.idx = idx
            self.forward_called = False
            
        def forward(self, hidden_state, *args, **kwargs):
            print(f"  Running layer {self.idx} (mock)")
            self.forward_called = True
            return hidden_state + 0.1 * np.random.randn(*hidden_state.shape)
    
    class MockModel:
        def __init__(self):
            self.config = MockConfig()
            self.layers = [MockLayer(i) for i in range(self.config.num_hidden_layers)]
            
        def __call__(self, input_ids=None, hidden_states=None):
            batch_size = 1
            seq_len = len(input_ids) if input_ids is not None else 10
            hidden_dim = self.config.hidden_size
            
            # Create mock hidden states
            if hidden_states is None:
                hidden_states = np.random.randn(batch_size, seq_len, hidden_dim)
                
            # Process through layers
            for layer in self.layers:
                hidden_states = layer.forward(hidden_states)
                
            return {"last_hidden_state": hidden_states}
    
    return MockModel()

def get_transformer_layers(model):
    """Extract the transformer layers from the model based on its architecture."""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        return model.transformer.h
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT style
        return model.encoder.layer
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Some other architectures
        return model.model.layers
    elif hasattr(model, 'layers'):
        # Direct layers
        return model.layers
    else:
        raise ValueError(f"Could not identify transformer layers in model of type {type(model).__name__}")

def run_example(args):
    print("\n=== Layer Compression and Skipping Example ===\n")
    
    if HAS_TORCH:
        print(f"Loading model {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Set input text
        input_text = "This is an example of using layer compression and skipping."
        print(f"Input text: '{input_text}'")
        
        # Tokenize input
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        print(f"Input sequence length: {input_ids.shape[1]}")
        
        # Determine model architecture
        try:
            transformer_layers = get_transformer_layers(model)
            num_layers = len(transformer_layers)
            print(f"Detected {num_layers} transformer layers")
            
            # Adjust model.config.num_hidden_layers if needed
            if not hasattr(model.config, 'num_hidden_layers') or model.config.num_hidden_layers != num_layers:
                print(f"Setting model.config.num_hidden_layers to {num_layers}")
                model.config.num_hidden_layers = num_layers
        except ValueError as e:
            print(f"Warning: {str(e)}")
            print("Using default model configuration")
    else:
        print("Creating mock model...")
        model = create_mock_model()
        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        print("Using mock input with 10 tokens")

    # Define which layers are in the hot path (never skipped)
    # Usually first few layers and last layer
    hot_path_indices = [0, 1, model.config.num_hidden_layers - 1]
    print(f"Hot path layers (never skipped): {hot_path_indices}")
    
    # Define the threshold for skipping
    skip_threshold = args.skip_threshold
    print(f"Skip threshold: {skip_threshold}")
    
    # Initialize the Layer Compression and Skipping manager
    lcs = LayerCompressionAndSkipping(
        model=model,
        compression_rank=args.compression_rank,
        hot_path_indices=hot_path_indices,
        skip_threshold=skip_threshold
    )
    
    # Apply low-rank factorization to the model
    print("\nApplying low-rank factorization...")
    lcs.apply_low_rank_factorization()
    
    # Add gating functions to determine which layers to skip
    print("\nAdding layer gating functions...")
    lcs.update_model_with_gating()
    
    # Run inference with dynamic layer skipping
    print("\nRunning inference with layer compression and skipping...")
    
    # Demonstrate inference with both regular and compressed/skipped models
    if HAS_TORCH:
        # First run without layer skipping for comparison
        print("\n1. Running model without layer skipping (baseline):")
        with torch.no_grad():
            start_time = time.time()
            original_outputs = model(input_ids)
            baseline_time = time.time() - start_time
            print(f"Baseline inference time: {baseline_time:.4f} seconds")
        
        # Adjust layer skipping threshold based on available compute
        print("\n2. Adjusting compression and skipping based on available compute:")
        available_compute = args.available_compute  # 1.0 = abundant resources, 0.0 = very limited
        print(f"Available compute factor: {available_compute}")
        lcs.adjust_compression_level(available_compute)
        
        # Now run with layer skipping
        print("\n3. Running model with layer compression and skipping:")
        
        # We need to create a function to process through the model manually
        # so we can see which layers are being skipped
        
        def process_with_layer_skipping(model, input_ids):
            """Process through the model with layer skipping."""
            # Determine model architecture and how to access layers
            try:
                transformer_layers = get_transformer_layers(model)
            except ValueError:
                print("Cannot access model layers directly, using standard inference")
                return model(input_ids)
                
            # Get embeddings (model-specific)
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                # GPT-2 style
                hidden_states = model.transformer.wte(input_ids)
                if hasattr(model.transformer, 'wpe'):
                    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                    hidden_states = hidden_states + model.transformer.wpe(position_ids)
            elif hasattr(model, 'get_input_embeddings'):
                # Generic transformer
                hidden_states = model.get_input_embeddings()(input_ids)
            else:
                print("Cannot determine how to get input embeddings, using standard inference")
                return model(input_ids)
                
            # Calculate temperatures for all layers
            lcs.compute_layer_temperatures(input_ids, hidden_states)
            
            # Process through each layer with skipping
            for i, layer in enumerate(transformer_layers):
                if lcs.should_skip_layer(i, hidden_states):
                    print(f"  Skipping layer {i}")
                else:
                    print(f"  Processing layer {i}")
                    # Different layer structures return different things
                    layer_output = layer(hidden_states)
                    
                    # Handle different return values
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output
            
            # Final normalization and head (model-specific)
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
                # GPT-2 style
                hidden_states = model.transformer.ln_f(hidden_states)
                lm_logits = model.lm_head(hidden_states)
                return {"logits": lm_logits, "last_hidden_state": hidden_states}
            elif hasattr(model, 'get_output_embeddings'):
                # Generic approach
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None:
                    logits = output_embeddings(hidden_states)
                    return {"logits": logits, "last_hidden_state": hidden_states}
            
            # Fallback
            return {"last_hidden_state": hidden_states}
        
        start_time = time.time()
        with torch.no_grad():
            try:
                outputs = process_with_layer_skipping(model, input_ids)
                skipped_inference_time = time.time() - start_time
                print(f"Layer-skipped inference time: {skipped_inference_time:.4f} seconds")
                print(f"Speedup: {baseline_time / max(0.001, skipped_inference_time):.2f}x")
            except Exception as e:
                print(f"Error during layer-skipped inference: {str(e)}")
                print("Falling back to simpler implementation...")
                
                # Simpler approach that doesn't directly observe layer skipping
                # Here we're using the gating functions added by update_model_with_gating
                start_time = time.time()
                outputs = model(input_ids)
                skipped_inference_time = time.time() - start_time
                print(f"Layer-skipped inference time: {skipped_inference_time:.4f} seconds")
                print(f"Speedup: {baseline_time / max(0.001, skipped_inference_time):.2f}x")
    else:
        # Mock implementation for non-PyTorch environments
        print("\nSimulating inference with mock model...")
        
        # Calculate layer temperatures using mock hidden states
        mock_hidden_states = np.random.randn(1, input_ids.shape[1], model.config.hidden_size)
        lcs.compute_layer_temperatures(input_ids, mock_hidden_states)
        
        # Simulate layer skipping
        for i in range(model.config.num_hidden_layers):
            should_skip = lcs.should_skip_layer(i, mock_hidden_states)
            if should_skip:
                print(f"  Layer {i}: SKIPPED")
            else:
                print(f"  Layer {i}: EXECUTED")
                # Update hidden states as if we executed this layer
                mock_hidden_states += 0.1 * np.random.randn(*mock_hidden_states.shape)
    
    # Get metrics
    metrics = lcs.get_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nExample completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Layer Compression and Skipping Example')
    parser.add_argument('--model-name', type=str, default='gpt2', 
                        help='HuggingFace model name (default: gpt2)')
    parser.add_argument('--compression-rank', type=int, default=8,
                        help='Rank for low-rank factorization (default: 8)')
    parser.add_argument('--skip-threshold', type=float, default=0.3,
                        help='Threshold for layer skipping (default: 0.3)')
    parser.add_argument('--available-compute', type=float, default=0.5,
                        help='Available compute factor 0.0-1.0 (default: 0.5)')
    
    args = parser.parse_args()
    run_example(args) 