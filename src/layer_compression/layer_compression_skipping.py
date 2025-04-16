import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union

import torch

# Check if torch is available (still useful for conditional logic)
HAS_TORCH = False
try:
    # Ensure torch was imported successfully
    if torch.__version__:
        HAS_TORCH = True
except (NameError, AttributeError):
    # torch might not be imported or might not have __version__
    HAS_TORCH = False

class LayerCompressionAndSkipping:
    """
    Layer Compression and Skipping Manager
    
    This class handles techniques to reduce the computational cost of running
    Transformer layers through compression and selective execution.
    
    Key features:
    - Applies low-rank factorization (LoRA-style) to reduce parameter size
    - Implements dynamic layer skipping using a gating function
    - Maintains a "hot path" of essential layers that are never skipped
    - Adapts to varying computational constraints at runtime
    
    Based on the third technique from the research paper on efficient
    on-device LLM inference.
    """
    
    def __init__(self, model, compression_rank=8, hot_path_indices=None, skip_threshold=0.3):
        """
        Initialize the Layer Compression and Skipping manager.
        
        Args:
            model: The Transformer model to optimize
            compression_rank: Rank for the low-rank factorization (lower = more compression)
            hot_path_indices: Indices of layers that should never be skipped
            skip_threshold: Threshold for the gating function to decide layer skipping
        """
        self.model = model
        self.compression_rank = compression_rank
        self.hot_path_indices = hot_path_indices or [0, 1, model.config.num_hidden_layers-1] if hasattr(model, 'config') else [0, 1]
        self.skip_threshold = skip_threshold
        self.layer_temperatures = {}  # Tracks "temperature" (importance) of each layer
        self.compressed_layers = {}   # Stores low-rank factorized weights
        
        # Initialize metrics
        self.metrics = {
            "layers_skipped": 0,
            "total_layers_evaluated": 0,
            "compression_ratio": 0.0,
            "skipping_efficiency": 0.0,
            "total_computation_saved": 0.0
        }
        
    def apply_low_rank_factorization(self, layer_indices=None):
        """
        Apply LoRA-style low-rank factorization to model layers.
        
        As described in the paper, we decompose large weight matrices W into:
        W = W₀ + BA, where B and A are low-rank matrices.
        
        Args:
            layer_indices: Specific layers to compress; if None, compress all
            
        Returns:
            Compressed model with factorized layers
        """ 
        # Get all layer indices if not specified
        if layer_indices is None:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
                layer_indices = list(range(num_layers))
            else:
                print("Warning: No layer indices provided and couldn't determine model structure.")
                return self.model
        
        # Default compression ratio (in case no layers are compressed)
        compression_ratio = 1.0
        total_original_params = 0
        total_compressed_params = 0
        
        # Apply factorization to specified layers
        for idx in layer_indices:
            # Skip hot path layers if we want to preserve their full accuracy
            if idx in self.hot_path_indices and idx != self.hot_path_indices[-1]:  # Still compress last hot path layer
                continue
                
            # Get the layer weights
            try:
                ratio = 1.0  # Default if no compression is applied
                
                if HAS_TORCH:
                    # Try to find layers in different model architectures
                    # GPT-2 style
                    if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                        if idx < len(self.model.transformer.h):
                            layer = self.model.transformer.h[idx]
                            
                            # Try to find weight matrices in the layer
                            weight_matrices = []
                            
                            # Find all weight matrices in this layer
                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_fc'):
                                weight_matrices.append((layer.mlp.c_fc, 'weight'))
                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_proj'):
                                weight_matrices.append((layer.mlp.c_proj, 'weight'))
                            if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
                                weight_matrices.append((layer.attn.c_attn, 'weight'))
                            if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_proj'):
                                weight_matrices.append((layer.attn.c_proj, 'weight'))
                            
                            # Process each weight matrix
                            layer_original_params = 0
                            layer_compressed_params = 0
                            
                            for module, weight_name in weight_matrices:
                                # Get the weight tensor
                                weight = getattr(module, weight_name)
                                weight_shape = weight.shape
                                weight_data = weight.data
                                
                                # Determine rank for this matrix
                                matrix_rank = min(self.compression_rank, min(weight_shape))
                                
                                # Perform SVD for actual compression
                                try:
                                    # Reshape if needed (e.g. for conv layers)
                                    if len(weight_shape) > 2:
                                        orig_shape = weight_shape
                                        weight_data = weight_data.reshape(weight_shape[0], -1)
                                    
                                    # Perform SVD
                                    U, S, V = torch.svd(weight_data)
                                    
                                    # Take only top-k singular values/vectors
                                    U_k = U[:, :matrix_rank]
                                    S_k = torch.diag(S[:matrix_rank])
                                    V_k = V[:, :matrix_rank]
                                    
                                    # Create the factorized matrices B and A
                                    B = U_k @ torch.sqrt(S_k)
                                    A = torch.sqrt(S_k) @ V_k.T
                                    
                                    # Store the original weight and its factorization
                                    self.compressed_layers[f"{idx}_{module.__class__.__name__}_{weight_name}"] = {
                                        'original_shape': weight_shape,
                                        'original_weight': weight_data.clone(),
                                        'B': B,
                                        'A': A,
                                        'module': module,
                                        'weight_name': weight_name
                                    }
                                    
                                    # Actually update the weights with the low-rank approximation
                                    # Reconstruct the approximated weight matrix
                                    approximated_weight = B @ A
                                    
                                    # Reshape back if needed
                                    if len(weight_shape) > 2:
                                        approximated_weight = approximated_weight.reshape(orig_shape)
                                    
                                    # Update the model weights
                                    with torch.no_grad():
                                        weight.copy_(approximated_weight)
                                    
                                    # Count parameters
                                    original_params = np.prod(weight_shape)
                                    compressed_params = matrix_rank * (weight_shape[0] + weight_shape[1])
                                    
                                    layer_original_params += original_params
                                    layer_compressed_params += compressed_params
                                    
                                except Exception as e:
                                    print(f"SVD failed for layer {idx}, module {module.__class__.__name__}: {str(e)}")
                                    continue
                            
                            if layer_original_params > 0:
                                layer_ratio = layer_compressed_params / layer_original_params
                                ratio = layer_ratio
                                
                                print(f"Layer {idx} compressed with ratio: {layer_ratio:.4f}")
                                print(f"  Original params: {layer_original_params}, Compressed params: {layer_compressed_params}")
                                
                                total_original_params += layer_original_params
                                total_compressed_params += layer_compressed_params
                    else:
                        # Try to find layers in other model structures
                        # For BERT style
                        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                            if idx < len(self.model.encoder.layer):
                                bert_layer = self.model.encoder.layer[idx]
                                
                                # Try to compress the attention and intermediate layers
                                weight_matrices = []
                                
                                if hasattr(bert_layer, 'attention'):
                                    if hasattr(bert_layer.attention.self, 'query'):
                                        weight_matrices.append((bert_layer.attention.self.query, 'weight'))
                                    if hasattr(bert_layer.attention.self, 'key'):
                                        weight_matrices.append((bert_layer.attention.self.key, 'weight'))
                                    if hasattr(bert_layer.attention.self, 'value'):
                                        weight_matrices.append((bert_layer.attention.self.value, 'weight'))
                                    if hasattr(bert_layer.attention, 'output') and hasattr(bert_layer.attention.output, 'dense'):
                                        weight_matrices.append((bert_layer.attention.output.dense, 'weight'))
                                
                                if hasattr(bert_layer, 'intermediate') and hasattr(bert_layer.intermediate, 'dense'):
                                    weight_matrices.append((bert_layer.intermediate.dense, 'weight'))
                                
                                if hasattr(bert_layer, 'output') and hasattr(bert_layer.output, 'dense'):
                                    weight_matrices.append((bert_layer.output.dense, 'weight'))
                                
                                # Process each weight matrix
                                layer_original_params = 0
                                layer_compressed_params = 0
                                
                                for module, weight_name in weight_matrices:
                                    # Get the weight tensor
                                    weight = getattr(module, weight_name)
                                    weight_shape = weight.shape
                                    weight_data = weight.data
                                    
                                    # Determine rank for this matrix
                                    matrix_rank = min(self.compression_rank, min(weight_shape))
                                    
                                    # Perform SVD for actual compression
                                    try:
                                        # Perform SVD
                                        U, S, V = torch.svd(weight_data)
                                        
                                        # Take only top-k singular values/vectors
                                        U_k = U[:, :matrix_rank]
                                        S_k = torch.diag(S[:matrix_rank])
                                        V_k = V[:, :matrix_rank]
                                        
                                        # Create the factorized matrices B and A
                                        B = U_k @ torch.sqrt(S_k)
                                        A = torch.sqrt(S_k) @ V_k.T
                                        
                                        # Store the original weight and its factorization
                                        self.compressed_layers[f"{idx}_{module.__class__.__name__}_{weight_name}"] = {
                                            'original_shape': weight_shape,
                                            'original_weight': weight_data.clone(),
                                            'B': B,
                                            'A': A,
                                            'module': module,
                                            'weight_name': weight_name
                                        }
                                        
                                        # Actually update the weights with the low-rank approximation
                                        # Reconstruct the approximated weight matrix
                                        approximated_weight = B @ A
                                        
                                        # Update the model weights
                                        with torch.no_grad():
                                            weight.copy_(approximated_weight)
                                        
                                        # Count parameters
                                        original_params = np.prod(weight_shape)
                                        compressed_params = matrix_rank * (weight_shape[0] + weight_shape[1])
                                        
                                        layer_original_params += original_params
                                        layer_compressed_params += compressed_params
                                        
                                    except Exception as e:
                                        print(f"SVD failed for BERT layer {idx}, module {module.__class__.__name__}: {str(e)}")
                                        continue
                                
                                if layer_original_params > 0:
                                    layer_ratio = layer_compressed_params / layer_original_params
                                    ratio = layer_ratio
                                    
                                    print(f"BERT Layer {idx} compressed with ratio: {layer_ratio:.4f}")
                                    print(f"  Original params: {layer_original_params}, Compressed params: {layer_compressed_params}")
                                    
                                    total_original_params += layer_original_params
                                    total_compressed_params += layer_compressed_params
                else:
                    # Numpy-based implementation
                    print(f"Using NumPy-based compression for layer {idx}")
                    
                    # This is a simplified example using random matrices
                    # In a real implementation, you would extract the actual model weights
                    shape = (512, 2048)  # Example shape
                    matrix_rank = min(self.compression_rank, min(shape))
                    
                    # Create a random matrix to compress
                    W = np.random.randn(*shape)
                    
                    # Perform SVD
                    U, S, Vt = np.linalg.svd(W, full_matrices=False)
                    
                    # Take only top-k singular values/vectors
                    U_k = U[:, :matrix_rank]
                    S_k = np.diag(S[:matrix_rank])
                    Vt_k = Vt[:matrix_rank, :]
                    
                    # Create the factorized matrices B and A
                    B = U_k @ np.sqrt(S_k)
                    A = np.sqrt(S_k) @ Vt_k
                    
                    # Store the factorization
                    self.compressed_layers[f"{idx}_mock"] = {
                        'original_shape': shape,
                        'B': B,
                        'A': A
                    }
                    
                    # Calculate compression ratio
                    original_params = np.prod(shape)
                    compressed_params = matrix_rank * (shape[0] + shape[1])
                    ratio = compressed_params / original_params
                    
                    print(f"NumPy Layer {idx} mock-compressed with ratio: {ratio:.4f}")
                    print(f"  Original params: {original_params}, Compressed params: {compressed_params}")
                    
                    total_original_params += original_params
                    total_compressed_params += compressed_params
                    
                # Update metrics with the last compression ratio calculated
                compression_ratio = ratio
            except Exception as e:
                print(f"Error compressing layer {idx}: {str(e)}")
        
        # Update overall compression metrics
        if total_original_params > 0:
            overall_ratio = total_compressed_params / total_original_params
            print(f"\nOverall compression ratio: {overall_ratio:.4f}")
            print(f"Total original parameters: {total_original_params:,}")
            print(f"Total compressed parameters: {total_compressed_params:,}")
            print(f"Memory savings: {(1 - overall_ratio) * 100:.2f}%")
            
            self.metrics["compression_ratio"] = overall_ratio
        else:
            self.metrics["compression_ratio"] = compression_ratio
                
        return self.model
        
    def _compute_layer_temperature(self, layer_idx: int, current_hidden_state: torch.Tensor) -> float:
        """
        Compute the "temperature" (importance) for a specific layer based on its input.
        
        Args:
            layer_idx: The index of the layer.
            current_hidden_state: The hidden state input to this specific layer.
            
        Returns:
            Temperature score (float between 0 and 1).
        """
        if not (HAS_TORCH and isinstance(current_hidden_state, torch.Tensor)):
            # Fallback for non-torch or missing input
            if layer_idx in self.hot_path_indices:
                return 0.9 # Default high for hot path
            else:
                # Simple random baseline if no tensor input
                return 0.1 + 0.6 * np.random.random()

        try:
            # Get number of layers if not already stored
            # This assumes self.model and self.model.config are available
            # It might be better to pass num_layers during init or store it more reliably
            if hasattr(self, 'num_layers') and self.num_layers > 0:
                num_layers = self.num_layers
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
                self.num_layers = num_layers # Cache it
            else:
                num_layers = 12  # Default assumption if config unavailable
                self.num_layers = num_layers # Cache it

            # 1. Activation magnitude (using the specific input to this layer)
            activation_magnitude = torch.mean(torch.abs(current_hidden_state)).item()
            
            # 2. Activation variance
            activation_variance = torch.var(current_hidden_state).item()
            
            # 3. Layer position importance (earlier and later layers are more important)
            # Avoid division by zero if num_layers is 1
            position_factor = 1.0
            if num_layers > 1:
                 position_factor = 1.0 - abs(layer_idx - (num_layers - 1) / 2.0) / ((num_layers - 1) / 2.0)
            
            # 4. Hot path bonus (should ideally not be needed here as should_skip_layer handles it)
            # hot_path_bonus = 0.2 if layer_idx in self.hot_path_indices else 0.0
            
            # Combine factors
            # Normalize components (simple clipping for demo)
            magnitude_norm = min(1.0, activation_magnitude / 10.0) # Assuming avg magnitude rarely exceeds 10?
            variance_norm = min(1.0, activation_variance / 1.0)    # Assuming variance rarely exceeds 1?
            
            # Weighted combination (Adjust weights as needed)
            layer_temp = (
                0.5 * magnitude_norm +      # Activation magnitude
                0.3 * variance_norm +       # Activation variance
                0.2 * position_factor       # Layer position
            )
            
            # Ensure temperature is in [0, 1] range
            layer_temp_clamped = max(0.0, min(1.0, layer_temp))
            
            # Detailed logging for temperature calculation
            # Note: This logging might become very verbose
            # Use a logger if available, otherwise print
            try:
                 # Basic print for now, assuming no logger is easily passed here
                 print(f"[DEBUG Temp Layer {layer_idx}] Raw Temp: {layer_temp:.4f} | Clamped: {layer_temp_clamped:.4f} | MagNorm: {magnitude_norm:.4f} (AvgMag: {activation_magnitude:.4f}) | VarNorm: {variance_norm:.4f} (Var: {activation_variance:.4f}) | PosF: {position_factor:.4f}")
            except Exception as log_e:
                 print(f"[DEBUG Temp Layer {layer_idx}] Logging error: {log_e}")
            # ----------------------------------------------------------

            return float(layer_temp_clamped) # Return the clamped value

        except Exception as e:
            print(f"Error computing temperature for layer {layer_idx}: {e}")
            # Return a default safe value (e.g., high temp to avoid skipping)
            return 0.9 
        
    def should_skip_layer(self, layer_idx, hidden_state):
        """
        Determine if a given layer should be skipped based on its temperature
        and whether it's part of the hot path.
        
        As described in the paper, we use a gating function g(·) that, given the 
        hidden state, decides whether to skip the layer or process it.
        
        Args:
            layer_idx: Index of the layer
            hidden_state: Input hidden state to the layer
            
        Returns:
            Boolean indicating whether to skip the layer
        """
        # Hot path layers are never skipped (Checked by caller now, but double-check here)
        if layer_idx in self.hot_path_indices:
            return False
            
        # Calculate temperature dynamically for this layer
        temperature = self._compute_layer_temperature(layer_idx, hidden_state)
        
        # Update metrics
        self.metrics["total_layers_evaluated"] += 1
        
        # The gating function: skip if temperature is below threshold
        if temperature < self.skip_threshold:
            # This layer will be skipped
            self.metrics["layers_skipped"] += 1
            # Avoid division by zero if no layers evaluated yet
            if self.metrics["total_layers_evaluated"] > 0:
                self.metrics["skipping_efficiency"] = self.metrics["layers_skipped"] / self.metrics["total_layers_evaluated"]
            return True
        else:
            return False
        
    def get_gating_function(self, layer_idx):
        """
        Create a gating function for a specific layer that decides whether
        to skip the layer at runtime.
        
        The gating function is a lightweight neural network that predicts
        whether a layer should be executed or skipped for the current input.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            A function that takes layer input and returns a skip decision
        """
        # Define a closure that uses our should_skip_layer method
        def gating_function(hidden_state):
            return self.should_skip_layer(layer_idx, hidden_state)
            
        return gating_function
        
    def run_layer_or_skip(self, layer_idx, layer_fn, hidden_state, *args, **kwargs):
        """
        Decide whether to run a layer or skip it based on temperature.
        
        This function implements the core decision logic for layer skipping.
        
        Args:
            layer_idx: Index of the layer
            layer_fn: Original layer forward function
            hidden_state: Current hidden state tensor
            args, kwargs: Additional arguments to pass to the layer
            
        Returns:
            Output from either running the layer or skipping it
        """
        import time
        
        # If layer is in the hot path, never skip it
        if layer_idx in self.hot_path_indices:
            print(f"Layer {layer_idx} is in hot path - executing (temperature: always executed)")
            start_time = time.time()
            result = layer_fn(hidden_state)
            duration = time.time() - start_time
            print(f"  Hot path layer {layer_idx} executed in {duration:.6f}s")
            return result
            
        # Get layer temperature (or default to 0.5 if not available)
        layer_temp = self.layer_temperatures.get(layer_idx, 0.5)
        
        # Update metrics
        self.metrics["total_layers_evaluated"] += 1
        
        # Skip the layer if temperature is below threshold
        if layer_temp < self.skip_threshold:
            self.metrics["layers_skipped"] += 1
            
            # Calculate skipping efficiency
            total_evaluated = self.metrics["total_layers_evaluated"]
            total_skipped = self.metrics["layers_skipped"]
            
            if total_evaluated > 0:
                self.metrics["skipping_efficiency"] = total_skipped / total_evaluated
                
            # For computation savings, we count each skipped layer (roughly)
            # as reducing computation by 1/num_layers for the current token
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
                self.metrics["total_computation_saved"] += 1.0 / num_layers
                
            print(f"Layer {layer_idx} skipped with temperature {layer_temp:.4f} (below threshold {self.skip_threshold:.2f})")
            
            # Return input as output when skipping
            # This is the key part of skipping - we don't transform the hidden state
            return hidden_state
        else:
            # Execute the layer normally
            start_time = time.time()
            result = layer_fn(hidden_state)
            duration = time.time() - start_time
            
            print(f"Layer {layer_idx} executed with temperature {layer_temp:.4f} in {duration:.6f}s")
            return result
        
    def update_model_with_gating(self):
        """
        Update the model by adding gating functions to each layer for dynamic skipping.
        
        This method modifies the forward pass of the model to include our 
        gating mechanism for layer skipping.
        
        Returns:
            Model with gating functions attached
        """
        # This implementation is heavily dependent on the model architecture
        # Here we provide a conceptual outline
        
        try:
            # Try different model architectures
            modified_layers = False
            num_layers = 0
            
            # Check for GPT-2 style architecture
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layers = self.model.transformer.h
                num_layers = len(layers)
                print(f"Detected GPT-2 style architecture with {num_layers} layers")
                modified_layers = self._add_gating_to_layers(layers)
            
            # Check for BERT style architecture
            elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                layers = self.model.encoder.layer
                num_layers = len(layers)
                print(f"Detected BERT style architecture with {num_layers} layers")
                modified_layers = self._add_gating_to_layers(layers)
            
            # Check for direct layers attribute
            elif hasattr(self.model, 'layers'):
                layers = self.model.layers
                num_layers = len(layers)
                print(f"Detected model with direct layers attribute ({num_layers} layers)")
                modified_layers = self._add_gating_to_layers(layers)
            
            # Check for custom layer structure
            elif hasattr(self.model, 'get_layer'):
                # Try to get number of layers from config
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                    num_layers = self.model.config.num_hidden_layers
                    print(f"Detected custom architecture with {num_layers} layers")
                    # Create a list of layer getters
                    layers = [lambda i=i: self.model.get_layer(i) for i in range(num_layers)]
                    modified_layers = self._add_gating_to_layers(layers)
            
            if modified_layers:
                print(f"\nSuccessfully added gating functions to {num_layers} layers")
                print(f"Hot path layers (never skipped): {self.hot_path_indices}")
                print(f"Skip threshold: {self.skip_threshold}")
            else:
                print("\nWarning: Model structure not supported for automatic gating")
                print("Please ensure the model has one of the following structures:")
                print("- GPT-2 style: model.transformer.h")
                print("- BERT style: model.encoder.layer")
                print("- Direct layers: model.layers")
                print("- Custom: model.get_layer()")
                
        except Exception as e:
            print(f"Error adding gating functions: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return self.model
    
    def _add_gating_to_layers(self, layers):
        """
        Add gating functions to a list of layers to enable dynamic skipping.
        
        This patches each layer's forward method with a new function that decides
        whether to execute the layer or skip it based on temperature.
        
        Args:
            layers: List of transformer layers to patch
            
        Returns:
            Boolean indicating if the layers were successfully modified
        """
        import types
        import functools
        
        if not layers:
            print("No layers provided to add gating functions.")
            return False
            
        print(f"Adding gating functions to {len(layers)} layers")
        
        # Track whether we modified any layers
        modified_layers = False
        
        # Keep a reference to self
        compression_manager = self
        
        for idx, layer in enumerate(layers):
            # Check if this layer has a forward method that we can patch
            if hasattr(layer, 'forward') and callable(layer.forward):
                # Store original forward method
                orig_forward = layer.forward
                
                # Create a new gated forward function that has access to both
                # the layer and the compression manager
                def make_gated_forward(idx, orig_forward, layer, manager):
                    def gated_forward(self, hidden_states=None, layer_past=None, attention_mask=None, 
                                     head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, 
                                     use_cache=False, output_attentions=False, *args, **kwargs):
                        """Gated forward that decides whether to execute or skip the layer."""
                        # Check if we have hidden states
                        if hidden_states is None:
                            print(f"Warning: Layer {idx} received None as hidden_states")
                            # Call original with same args
                            return orig_forward(hidden_states=hidden_states, layer_past=layer_past, 
                                               attention_mask=attention_mask, head_mask=head_mask,
                                               encoder_hidden_states=encoder_hidden_states, 
                                               encoder_attention_mask=encoder_attention_mask,
                                               use_cache=use_cache, output_attentions=output_attentions,
                                               *args, **kwargs)
                        
                        # Use manager's run_layer_or_skip function to decide execution
                        def execute_layer(hidden_states):
                            return orig_forward(hidden_states=hidden_states, layer_past=layer_past,
                                              attention_mask=attention_mask, head_mask=head_mask,
                                              encoder_hidden_states=encoder_hidden_states,
                                              encoder_attention_mask=encoder_attention_mask,
                                              use_cache=use_cache, output_attentions=output_attentions,
                                              *args, **kwargs)
                        
                        # Use the manager's method to run or skip the layer
                        return manager.run_layer_or_skip(
                            layer_idx=idx,
                            layer_fn=execute_layer,
                            hidden_state=hidden_states
                        )
                    
                    return gated_forward
                
                # Bind the new method to the layer instance
                bound_forward = make_gated_forward(idx, orig_forward, layer, compression_manager)
                layer.forward = types.MethodType(bound_forward, layer)
                
                # Also store reference to compression manager for direct access
                layer._compression_manager = compression_manager
                
                print(f"  Added gating function to layer {idx}")
                modified_layers = True
            else:
                print(f"  Warning: Layer {idx} doesn't have a forward method that can be patched")
                
        if modified_layers:
            print(f"Successfully added gating functions to layers")
        else:
            print("No layers were modified with gating functions")
            
        return modified_layers
    
    def adjust_compression_level(self, available_compute):
        """
        Dynamically adjust compression level based on available compute resources.
        
        This allows the model to adapt to changing device conditions, increasing
        compression when resources are scarce.
        
        Args:
            available_compute: Measure of available computational resources
            
        Returns:
            Updated compression configuration
        """
        # Scale compression rank based on available compute
        # Lower available_compute means more aggressive compression
        
        # available_compute should be normalized between 0 and 1
        # where 0 means very limited resources and 1 means abundant resources
        available_compute = max(0.1, min(1.0, available_compute))
        
        # Adjust compression rank
        # Start with a minimum rank of 2
        min_rank = 2
        max_rank = 32
        
        # Linear scaling between min and max rank based on available compute
        new_rank = int(min_rank + (max_rank - min_rank) * available_compute)
        
        # Only update if significantly different
        if abs(new_rank - self.compression_rank) >= 2:
            old_rank = self.compression_rank
            self.compression_rank = new_rank
            print(f"Adjusted compression rank from {old_rank} to {new_rank} based on compute availability")
            
            # Re-apply compression if needed
            # This would be expensive, so in practice would need careful consideration
            if new_rank < old_rank:
                print("Re-compressing layers with new rank...")
                self.apply_low_rank_factorization()
                
        # Adjust skipping threshold too
        # Lower available compute = more aggressive skipping
        new_threshold = 0.3 + 0.4 * (1.0 - available_compute)
        
        if abs(new_threshold - self.skip_threshold) >= 0.05:
            old_threshold = self.skip_threshold
            self.skip_threshold = new_threshold
            print(f"Adjusted skip threshold from {old_threshold:.2f} to {new_threshold:.2f}")
            
        return {
            "compression_rank": self.compression_rank,
            "skip_threshold": self.skip_threshold
        }
        
    def get_metrics(self):
        """
        Return metrics about compression and layer skipping performance.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate total computation saved
        skipped_ratio = self.metrics["layers_skipped"] / max(1, self.metrics["total_layers_evaluated"])
        compressed_ratio = self.metrics["compression_ratio"]
        
        # Simplified computation saving estimate
        # Real implementation would be more sophisticated
        self.metrics["total_computation_saved"] = (
            skipped_ratio * 0.7 +  # Complete skipping saves 100% of that layer
            (1 - skipped_ratio) * compressed_ratio * 0.3  # Compression saves proportional to ratio
        )
        
        return self.metrics 