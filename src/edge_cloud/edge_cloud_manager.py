class EdgeCloudManager:
    """
    Edge-Cloud Collaborative Inference Manager
    
    This class handles dynamic partitioning of Transformer layers between local 
    hardware (CPU/GPU/NPU) and remote cloud servers based on runtime conditions.
    
    Key features:
    - Dynamically splits model layers between edge and cloud
    - Monitors network bandwidth and device load to optimize partitioning
    - Supports using a local mini-LLM for "easy" tokens
    - Includes optional encryption and dimensionality reduction for privacy
    
    Based on the first technique from the research paper on efficient 
    on-device LLM inference.
    """
    
    def __init__(self, 
                 model=None,
                 device_monitor=None, 
                 cloud_client=None, 
                 mini_llm=None,
                 privacy_protection=None, 
                 energy_weight=0.3, 
                 latency_weight=0.4, 
                 memory_weight=0.3):
        """
        Initialize the Edge-Cloud Manager.
        
        Args:
            model: The model to be partitioned
            device_monitor: Device monitoring component to track system resources
            cloud_client: Client for communicating with the cloud service
            mini_llm: Optional smaller model for handling "easy" tokens locally
            privacy_protection: Privacy protection component for securing data
            energy_weight: Importance of energy savings in partitioning decisions
            latency_weight: Importance of latency in partitioning decisions
            memory_weight: Importance of memory usage in partitioning decisions
        """
        self.model = model
        self.device_monitor = device_monitor
        self.cloud_client = cloud_client
        self.mini_llm = mini_llm
        self.privacy_protection = privacy_protection
        
        # Cost function weights
        self.energy_weight = energy_weight
        self.latency_weight = latency_weight
        self.memory_weight = memory_weight
        
        # Current partition strategy
        self.current_partition = {}  # Maps layer_idx -> "local" or "remote"
        self.pending_requests = {}   # Maps request_ids to metadata about pending cloud requests
        
        # Performance metrics
        self.metrics = {
            "cloud_requests": 0,
            "local_inferences": 0,
            "avg_latency_ms": 0,
            "mini_llm_usage": 0,
            "total_tokens": 0
        }
        
    def measure_network_conditions(self):
        """
        Measure current network bandwidth and latency.
        
        Returns:
            dict: Network metrics including bandwidth and latency
        """
        if self.device_monitor:
            metrics = self.device_monitor.get_metrics()
            return metrics["network"]
            
        # Fallback to default values if no device monitor provided
        return {"bandwidth_mbps": 1.0, "latency_ms": 100, "connected": True}
        
    def measure_device_load(self):
        """
        Measure current device CPU/GPU/memory utilization.
        
        Returns:
            dict: Device load metrics
        """
        if self.device_monitor:
            metrics = self.device_monitor.get_metrics()
            return metrics["hardware"]
            
        # Fallback to default values if no device monitor provided
        return {"cpu_usage_percent": 50, "memory": {"available_mb": 1000, "used_percent": 50}}
        
    def decide_partition(self, layer_idx, local_cost, remote_cost):
        """
        Determine whether a layer should be executed locally or remotely.
        
        As described in the paper, this implements a simple cost model C(S) for 
        comparing local vs. remote execution costs, factoring in computation,
        communication, and energy considerations.
        
        Args:
            layer_idx: Index of the Transformer layer
            local_cost: Estimated cost of local execution
            remote_cost: Estimated cost of remote execution
            
        Returns:
            str: "local" or "remote" decision
        """
        # Simple rule as specified: if local cost is lower, execute locally
        if local_cost < remote_cost:
            decision = "local"
        else:
            decision = "remote"
            
        # Store the decision for this layer
        self.current_partition[layer_idx] = decision
        return decision
        
    def determine_full_partition(self, model_config, input_length):
        """
        Determine the optimal partition for all layers based on current conditions.
        
        This implements the paper's approach for dynamic layer partitioning by
        estimating costs for each layer and deciding where to execute it.
        
        Args:
            model_config: Configuration of the Transformer model
            input_length: Current sequence length
            
        Returns:
            dict: Mapping from layer indices to "local" or "remote" decisions
        """
        network_metrics = self.measure_network_conditions()
        device_metrics = self.measure_device_load()
        
        # Reset partition decisions
        self.current_partition = {}
        
        # If not connected, run everything locally
        if not network_metrics.get("connected", False):
            num_layers = model_config.get("num_hidden_layers", 12)
            for layer_idx in range(num_layers):
                self.current_partition[layer_idx] = "local"
            return self.current_partition
        
        # For each layer, estimate costs and decide where to execute
        num_layers = model_config.get("num_hidden_layers", 12)
        hidden_size = model_config.get("hidden_size", 768)
        
        # Define base computation costs (estimated FLOPs per token per layer)
        # For a Transformer layer: 4 * hidden_size^2 (for each MLP matrix)
        # Plus 4 * hidden_size * input_length (for attention computation)
        base_compute_cost = 4 * hidden_size * hidden_size + 4 * hidden_size * input_length
        
        # Available memory on device (in MB)
        available_memory = device_metrics.get("memory", {}).get("available_mb", 1000)
        
        # Get the cloud cost multiplier based on bandwidth
        # Lower bandwidth = higher cloud cost
        bandwidth = network_metrics.get("bandwidth_mbps", 1.0)
        cloud_bandwidth_multiplier = max(0.1, min(1.0, 5.0 / bandwidth))
        
        # Get device load factor
        cpu_usage = device_metrics.get("cpu_usage_percent", 50) / 100.0
        device_load_factor = 1.0 + cpu_usage  # Higher CPU usage = higher local cost
        
        for layer_idx in range(num_layers):
            # Computation cost (higher for later layers due to complexity)
            layer_compute_factor = 1.0 + 0.1 * layer_idx  # Later layers are typically more complex
            
            # Memory cost (activation size for this layer)
            # Increase with sequence length and layer index (later layers might have larger activations)
            memory_cost = hidden_size * input_length * (1.0 + 0.05 * layer_idx) / 1024.0  # In MB
            
            # Check if we have enough memory for this layer
            memory_available_for_layer = available_memory - sum(
                hidden_size * input_length * (1.0 + 0.05 * idx) / 1024.0 
                for idx, decision in self.current_partition.items() 
                if decision == "local"
            )
            
            # If not enough memory, force cloud execution
            if memory_cost > memory_available_for_layer:
                self.current_partition[layer_idx] = "remote"
                continue
                
            # Calculate local execution cost
            local_compute_cost = base_compute_cost * layer_compute_factor * device_load_factor
            local_memory_cost = memory_cost / available_memory * 100  # Weight by percentage of memory used
            local_energy_cost = local_compute_cost * 0.01  # Simplified energy model
            
            total_local_cost = (
                self.latency_weight * local_compute_cost +
                self.memory_weight * local_memory_cost +
                self.energy_weight * local_energy_cost
            )
            
            # Calculate remote execution cost
            remote_compute_cost = base_compute_cost * 0.5  # Cloud typically has more compute power
            remote_memory_cost = 0  # Cloud has effectively unlimited memory for our purposes
            
            # Communication cost (data that needs to be transferred)
            # For sending to cloud: hidden_size * input_length bytes
            # For receiving from cloud: hidden_size * input_length bytes (simplified)
            communication_size_mb = 2 * hidden_size * input_length / (1024 * 1024)  # Convert to MB
            communication_cost = communication_size_mb / bandwidth * 1000  # Convert to ms
            
            # Remote energy cost (primarily from radio usage for communication)
            remote_energy_cost = communication_size_mb * 0.05  # Simplified energy model for communication
            
            total_remote_cost = (
                self.latency_weight * (remote_compute_cost + communication_cost) +
                self.memory_weight * remote_memory_cost +
                self.energy_weight * remote_energy_cost
            ) * cloud_bandwidth_multiplier
            
            # Make the decision
            self.decide_partition(layer_idx, total_local_cost, total_remote_cost)
            
        return self.current_partition
        
    def compress_and_encrypt(self, hidden_state):
        """
        Apply dimensionality reduction and optional encryption before sending to cloud.
        
        This implements the privacy protection approach described in the paper,
        which can include compression of intermediate states and encryption.
        
        Args:
            hidden_state: The intermediate activation tensors
            
        Returns:
            Compressed and/or encrypted state
        """
        if self.privacy_protection:
            return self.privacy_protection.protect(hidden_state)
            
        # Fallback to simple behavior if no privacy protection configured
        return {"tensor": hidden_state}
        
    def encrypt_intermediate_state(self, state):
        """
        Optionally encrypt the intermediate state before sending to cloud.
        
        Args:
            state: The intermediate activation tensors to encrypt
            
        Returns:
            Encrypted state
        """
        if self.privacy_protection and hasattr(self.privacy_protection, 'encryptor'):
            return self.privacy_protection.encryptor.encrypt_tensor(state)
            
        # No encryption available, return as is
        return state
        
    def reduce_state_dimensionality(self, state):
        """
        Apply dimensionality reduction to the intermediate state to reduce
        communication costs.
        
        As described in the paper, this reduces the size of data transferred
        to the cloud, which both improves privacy and reduces bandwidth usage.
        
        Args:
            state: The intermediate activation tensors
            
        Returns:
            Compressed state with reduced dimensions
        """
        if self.privacy_protection and hasattr(self.privacy_protection, 'reducer'):
            reduced, metadata = self.privacy_protection.reducer.reduce(state)
            return {"tensor": reduced, "metadata": metadata}
            
        # No reduction available, return as is
        return state
        
    def send_to_cloud(self, hidden_state, layer_idx=None, metadata=None):
        """
        Send intermediate state to cloud for further processing.
        
        Implements the cloud offloading mechanism described in the paper,
        with optional privacy protection.
        
        Args:
            hidden_state: The model's intermediate activations
            layer_idx: Index of the layer this state is from
            metadata: Additional metadata to include
            
        Returns:
            Request ID for tracking the cloud computation
        """
        # Apply privacy protection if configured
        if self.privacy_protection:
            protected_data = self.compress_and_encrypt(hidden_state)
        else:
            protected_data = {"tensor": hidden_state}
            
        # Add metadata
        if metadata:
            protected_data["metadata"] = metadata
        if layer_idx is not None:
            protected_data["layer_idx"] = layer_idx
            
        # Send to cloud using the cloud client
        if self.cloud_client:
            try:
                # Extract the actual tensor from the protected data
                if "tensor" in protected_data:
                    tensor = protected_data.pop("tensor")
                    request_id = self.cloud_client.send_tensor(
                        tensor=tensor,
                        layer_idx=layer_idx or 0,
                        metadata=protected_data
                    )
                else:
                    # If using full encryption, the tensor is in encrypted_package
                    request_id = self.cloud_client.send_tensor(
                        tensor=protected_data,
                        layer_idx=layer_idx or 0,
                        metadata={"fully_encrypted": True}
                    )
                
                # Store request for later
                self.pending_requests[request_id] = {
                    "timestamp": import time; time.time(),
                    "layer_idx": layer_idx,
                    "status": "pending"
                }
                
                # Update metrics
                self.metrics["cloud_requests"] += 1
                
                return request_id
                
            except Exception as e:
                # Log the error and fall back to local processing
                import logging
                logging.error(f"Failed to send data to cloud: {str(e)}")
                # If cloud fails, process locally as fallback
                return None
        else:
            # No cloud client configured, return a mock request ID
            import random
            request_id = f"mock-{random.randint(1000, 9999)}"
            return request_id
    
    def receive_from_cloud(self, request_id=None, wait=True):
        """
        Receive processed results from the cloud.
        
        Args:
            request_id: ID of the request to retrieve
            wait: Whether to wait for results if not immediately available
            
        Returns:
            Processed results from cloud
        """
        if self.cloud_client and request_id:
            try:
                result = self.cloud_client.get_result(request_id, wait=wait)
                
                # If result contains an encrypted tensor, decrypt it
                if self.privacy_protection and "encrypted_package" in result:
                    result["tensor"] = self.privacy_protection.unprotect(result)
                    
                # Update request status
                if request_id in self.pending_requests:
                    self.pending_requests[request_id]["status"] = "completed"
                    
                return result
                
            except Exception as e:
                # Log the error and return a mock result
                import logging
                logging.error(f"Failed to receive data from cloud: {str(e)}")
                return {"status": "error", "error": str(e)}
        else:
            # No cloud client configured, return mock data
            return {"status": "completed", "mock_result": "cloud_processed_data"}
    
    def is_token_easy(self, token_id, context):
        """
        Determine if a token is "easy" enough to be handled by mini-LLM.
        
        This implements the paper's approach for using a smaller local model
        for tokens that are predictable or less complex.
        
        Args:
            token_id: The token to evaluate
            context: Current context window
            
        Returns:
            Boolean indicating if the mini-LLM should handle this token
        """
        if self.mini_llm is None:
            return False
            
        # Use the mini-LLM's built-in decision logic
        return self.mini_llm.should_use_mini_llm(context, token_id)
    
    def process_locally(self, input_data, layer_indices=None):
        """
        Process input using only the local model.
        
        Args:
            input_data: Input token IDs or embeddings
            layer_indices: Specific layers to run locally (if None, run all)
            
        Returns:
            Model outputs from local device
        """
        if self.model is None:
            raise ValueError("No local model configured")
            
        # Create a subset of the model with only the specified layers
        import torch
        
        # Update metrics
        self.metrics["local_inferences"] += 1
        
        # If mini-LLM should be used for this token
        if self.is_token_easy(input_data, None):
            # Use mini-LLM for prediction
            self.metrics["mini_llm_usage"] += 1
            return self.mini_llm.predict_next_token(input_data)
            
        # Use the main model
        # In a real implementation, this would run only the specified layers
        # For now, we just run the full model
        with torch.no_grad():
            outputs = self.model(input_data)
            return outputs
            
    def process_hybrid(self, input_data):
        """
        Process input using a hybrid of local and cloud execution.
        
        This implements the core edge-cloud collaborative inference approach,
        dynamically splitting the model between local and cloud execution.
        
        Args:
            input_data: Input token IDs or embeddings
            
        Returns:
            Model outputs
        """
        # Determine the optimal layer partitioning
        input_length = len(input_data[0]) if isinstance(input_data, list) else input_data.shape[1]
        model_config = self.model.config if self.model else {"num_hidden_layers": 12, "hidden_size": 768}
        partition = self.determine_full_partition(model_config, input_length)
        
        # Process the input through the model, layer by layer
        import torch
        
        # Initialize with the input embedding
        if self.model:
            with torch.no_grad():
                # Get the input embeddings
                if hasattr(self.model, "get_input_embeddings"):
                    embeddings = self.model.get_input_embeddings()(input_data)
                else:
                    # Fall back to first layer of the model
                    embeddings = self.model.layers[0](input_data)
                    
                current_state = embeddings
                
                # Process each layer according to the partition
                for layer_idx, location in partition.items():
                    if location == "local":
                        # Process this layer locally
                        layer = self.model.layers[layer_idx]
                        current_state = layer(current_state)
                    else:
                        # Send to cloud for processing
                        request_id = self.send_to_cloud(current_state, layer_idx)
                        if request_id:
                            # Wait for result from cloud
                            result = self.receive_from_cloud(request_id)
                            if "tensor" in result:
                                current_state = result["tensor"]
                            else:
                                # If cloud processing failed, fall back to local
                                layer = self.model.layers[layer_idx]
                                current_state = layer(current_state)
                                
                # Final output processing
                if hasattr(self.model, "get_output"):
                    output = self.model.get_output(current_state)
                else:
                    # Fall back to simple output
                    output = current_state
                    
                return output
        else:
            # If no model, return mock output
            return {"mock_output": "hybrid_inference_result"} 