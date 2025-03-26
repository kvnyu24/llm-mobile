import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import torch

logger = logging.getLogger("edge_cloud_manager")

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
            "failed_cloud_requests": 0,
            "avg_latency_ms": 0,
            "mini_llm_usage": 0,
            "total_tokens": 0,
            "bandwidth_savings": 0,  # in MB
            "energy_savings": 0,     # estimated in Joules
        }
        
        # Execution mode can be forced for testing
        self.force_execution_mode = None  # None, "local", or "remote"
        
        # Cache for model layers and network statistics
        self.layer_costs = {}  # Cache for layer-wise costs
        self.last_partition = None  # Last partition decision
        
        # Start device monitoring if device monitor is provided and not already monitoring
        if self.device_monitor is not None and hasattr(self.device_monitor, 'is_monitoring'):
            if not self.device_monitor.is_monitoring():
                self.device_monitor.start_background_monitoring()
        
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
        
    def decide_partition(self, input_size, prompt_len=0):
        """
        Decide which layers to run locally vs in the cloud.
        
        Args:
            input_size: Size of the input tensor
            prompt_len: Length of the prompt (for autoregressive generation)
            
        Returns:
            A tuple (local_layers, remote_layers) indicating which layers should be run where
        """
        # Get number of layers from model config, or use default
        num_layers = 12  # Default number of layers
        if self.model is not None and hasattr(self.model, 'config'):
            num_layers = getattr(self.model.config, "num_hidden_layers", num_layers)
            
        # If we're forcing a specific execution mode, respect that
        if self.force_execution_mode == "local":
            logger.info("Forcing local execution for all layers")
            return list(range(num_layers)), []
        elif self.force_execution_mode == "remote":
            logger.info("Forcing remote execution for all layers")
            return [], list(range(num_layers))
            
        # Get current hardware and network conditions
        network_conditions = self.measure_network_conditions()
        device_load = self.measure_device_load()
        
        # No remote execution if network is down
        if network_conditions['bandwidth_mbps'] < 0.1:  # Very low bandwidth
            logger.warning("Network bandwidth too low, forcing local execution")
            return self.determine_full_partition(local_only=True)
        
        # Estimate costs for different partitioning strategies
        partitioning_options = self._generate_partitioning_options()
        lowest_cost = float('inf')
        best_partition = None
        
        for partition in partitioning_options:
            cost = self._estimate_partition_cost(
                partition, 
                network_conditions, 
                device_load, 
                input_size
            )
            
            if cost < lowest_cost:
                lowest_cost = cost
                best_partition = partition
        
        # Convert the partition representation to lists of local and remote layers
        local_layers, remote_layers = self._partition_to_layer_lists(best_partition)
        
        # Cache this decision
        self.last_partition = (local_layers, remote_layers)
        
        logger.info(f"Decided partition: {len(local_layers)} layers local, {len(remote_layers)} layers remote")
        return local_layers, remote_layers
        
    def _estimate_partition_cost(self, partition, network_conditions, device_load, input_size):
        """
        Estimate the cost of a given partitioning strategy.
        
        Args:
            partition: The partition to evaluate
            network_conditions: Current network metrics
            device_load: Current device metrics
            input_size: Size of the input tensor
            
        Returns:
            Estimated cost (lower is better)
        """
        # Simplified cost model
        num_layers = len(partition)
        local_layers = sum(1 for p in partition if p == "local")
        remote_layers = num_layers - local_layers
        
        # Base costs
        energy_cost = local_layers * 1.0  # Local execution costs energy
        latency_cost = 0.0
        memory_cost = local_layers * 1.0  # Local execution uses memory
        
        # Network-dependent costs
        if remote_layers > 0:
            # Communication cost
            bandwidth = network_conditions['bandwidth_mbps']
            if bandwidth < 0.1:
                bandwidth = 0.1  # Avoid division by zero
                
            # Higher bandwidth means lower latency cost
            transmission_latency = input_size / (bandwidth * 1024 * 1024 / 8)  # Size in bytes / bandwidth in bytes/sec
            
            # Cloud processing latency (assume constant for simplicity)
            cloud_latency = 0.5  # Default latency in seconds
            if self.cloud_client is not None and hasattr(self.cloud_client, 'latency'):
                cloud_latency = self.cloud_client.latency
            cloud_latency = cloud_latency * remote_layers
            
            # Total latency
            latency_cost = transmission_latency + cloud_latency
            
            # Energy cost of transmission
            transmission_energy = 0.5 * transmission_latency  # Simplified model
            energy_cost += transmission_energy
        
        # Weight the costs
        total_cost = (
            self.energy_weight * energy_cost +
            self.latency_weight * latency_cost +
            self.memory_weight * memory_cost
        )
        
        return total_cost
        
    def determine_full_partition(self, local_only=False):
        """
        Determine the full partitioning strategy for all layers.
        
        Args:
            local_only: Whether to force all computation to be local
            
        Returns:
            A tuple (local_layers, remote_layers) with layer indices
        """
        # Get number of layers from model config, or use default
        num_layers = 12  # Default number of layers
        if self.model is not None and hasattr(self.model, 'config'):
            num_layers = getattr(self.model.config, "num_hidden_layers", num_layers)
        
        if local_only or self.force_execution_mode == "local":
            # All layers local
            return list(range(num_layers)), []
        
        if self.force_execution_mode == "remote":
            # All layers remote
            return [], list(range(num_layers))
        
        # Get current network conditions
        network_conditions = self.measure_network_conditions()
        device_load = self.measure_device_load()
        
        # Estimate input size (this would be more accurate in a real implementation)
        input_size = 1024 * 1024  # 1MB as a placeholder
        
        # Use the decision method
        return self.decide_partition(input_size)
        
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
                import time
                self.pending_requests[request_id] = {
                    "timestamp": time.time(),
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
            
        # Convert context to tensor if it's not already
        if not isinstance(context, torch.Tensor):
            # If context is a list, convert to tensor
            if isinstance(context, list):
                context = torch.tensor([context], dtype=torch.long)
            # If context is a single int, make it a tensor with batch dimension
            elif isinstance(context, int):
                context = torch.tensor([[context]], dtype=torch.long)
            
        # Use the mini-LLM's built-in decision logic
        return self.mini_llm.should_use_mini_llm(context, None)
    
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
            
    def process_hybrid(self, input_ids):
        """
        Process the input with dynamic layer partitioning.
        
        This is the main entry point for the Edge-Cloud Collaborative Inference.
        It determines which layers to execute locally vs. remotely based on current conditions.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Model output with completed inference
        """
        # Get model config
        model_config = self.model.config
        input_length = input_ids.shape[1]
        
        # Determine layer partitioning
        local_layers, remote_layers = self.determine_full_partition()
        
        # For simulation, we'll run the full model if it's available 
        # and log which layers would be on device vs cloud
        if self.model is not None:
            logger.info(f"Running model inference with {len(local_layers)} local layers and {len(remote_layers)} remote layers")
            
            # In a real implementation, we would handle this by executing layers selectively
            # and transferring intermediate results between device and cloud
            
            # Log the layer partitioning for demonstration
            for layer_idx in range(model_config.num_hidden_layers):
                if layer_idx in local_layers:
                    logger.info(f"Layer {layer_idx} would run locally")
                else:
                    logger.info(f"Layer {layer_idx} would run remotely (in cloud)")
                
            # Run the model
            with torch.no_grad():
                outputs = self.model(input_ids)
                
            # Increment the metrics counter
            self.metrics["local_inferences"] += 1
            if remote_layers:  # If any layers were remote
                self.metrics["cloud_requests"] += 1
                
            return outputs
        
        # If no model is available, return a mock output
        return None
    
    def _partition_to_layer_lists(self, best_partition):
        """
        Convert the partition representation to lists of local and remote layers.
        
        Args:
            best_partition: The partition representation
            
        Returns:
            (local_layers, remote_layers): Lists of layer indices
        """
        # Simple implementation: assuming best_partition is already a list of "local" or "remote" decisions
        local_layers = []
        remote_layers = []
        
        for layer_idx, decision in enumerate(best_partition):
            if decision == "local":
                local_layers.append(layer_idx)
            else:
                remote_layers.append(layer_idx)
                
        return local_layers, remote_layers
        
    def _generate_partitioning_options(self):
        """
        Generate different partitioning strategies to evaluate.
        
        Returns:
            List of partitioning options to evaluate
        """
        # Get number of layers from model config, or use default
        num_layers = 12  # Default number of layers
        if self.model is not None and hasattr(self.model, 'config'):
            num_layers = getattr(self.model.config, "num_hidden_layers", num_layers)
        
        # For simplicity, we'll consider a few basic partitioning strategies:
        # 1. All local
        # 2. All remote
        # 3. First half local, second half remote
        # 4. First quarter local, rest remote
        # 5. First three quarters local, last quarter remote
        
        options = [
            ["local"] * num_layers,  # All local
            ["remote"] * num_layers,  # All remote
            ["local"] * (num_layers // 2) + ["remote"] * (num_layers - num_layers // 2),  # Half and half
            ["local"] * (num_layers // 4) + ["remote"] * (num_layers - num_layers // 4),  # Quarter local
            ["local"] * (3 * num_layers // 4) + ["remote"] * (num_layers - 3 * num_layers // 4)  # Three quarters local
        ]
        
        return options