"""
Cloud Client for Edge-Cloud Collaborative Inference

This module provides functionality to communicate with cloud services
for offloading model inference computation.
"""

import json
import time
import requests
import logging
from typing import Dict, Any, Optional, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cloud_client")

class CloudAPIClient:
    """
    Client for communicating with a cloud inference API.
    
    This handles sending intermediate model states to the cloud and
    retrieving processed results.
    """
    
    def __init__(
        self, 
        api_endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True
    ):
        """
        Initialize the Cloud API Client.
        
        Args:
            api_endpoint: URL of the cloud inference API
            api_key: Authentication key for the API (if required)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        
        # Configure headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
            
        # Keep track of in-flight requests
        self.pending_requests = {}
        
    def send_tensor(self, tensor: np.ndarray, layer_idx: int, metadata: Dict[str, Any] = None) -> str:
        """
        Send a tensor to the cloud API.
        
        Args:
            tensor: NumPy array to send
            layer_idx: Index of the model layer this tensor is from
            metadata: Additional information about the tensor/request
            
        Returns:
            Request ID for tracking the computation
        """
        # Convert tensor to a list for JSON serialization
        tensor_data = tensor.tolist()
        
        # Prepare payload
        payload = {
            "tensor": tensor_data,
            "layer_idx": layer_idx,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Send to API
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.api_endpoint}/process",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()
                result = response.json()
                request_id = result.get("request_id")
                
                # Store request ID for later retrieval
                if request_id:
                    self.pending_requests[request_id] = {
                        "status": "pending",
                        "timestamp": time.time(),
                        "layer_idx": layer_idx
                    }
                
                logger.info(f"Successfully sent tensor to cloud API. Request ID: {request_id}")
                return request_id
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                
        # If we get here, all retries failed
        raise RuntimeError("Failed to send tensor to cloud after maximum retries")
                
    def get_result(self, request_id: str, wait: bool = True, poll_interval: float = 0.5) -> Dict[str, Any]:
        """
        Retrieve results from the cloud API.
        
        Args:
            request_id: ID of the request to retrieve
            wait: Whether to wait for results if not immediately available
            poll_interval: How often to check for results when waiting
            
        Returns:
            Processed results from cloud
        """
        # Check if this is a known request
        if request_id not in self.pending_requests:
            logger.warning(f"Unknown request ID: {request_id}")
            
        # Try to retrieve the result
        url = f"{self.api_endpoint}/results/{request_id}"
        
        start_time = time.time()
        while True:
            try:
                response = self.session.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()
                result = response.json()
                
                # Check if processing is complete
                status = result.get("status")
                if status == "completed":
                    # Convert result tensor back to numpy if present
                    if "tensor" in result:
                        tensor_data = result["tensor"]
                        shape = result.get("shape")
                        dtype = result.get("dtype", "float32")
                        result["tensor"] = np.array(tensor_data, dtype=dtype).reshape(shape)
                    
                    # Update request status
                    if request_id in self.pending_requests:
                        self.pending_requests[request_id]["status"] = "completed"
                    
                    logger.info(f"Successfully retrieved result for request {request_id}")
                    return result
                elif status == "pending" or status == "processing":
                    if not wait:
                        return {"status": status, "message": "Processing not complete"}
                    
                    # Check if we've waited too long
                    elapsed = time.time() - start_time
                    if elapsed > self.timeout:
                        raise TimeoutError(f"Timed out waiting for result after {elapsed:.1f}s")
                    
                    # Wait and try again
                    time.sleep(poll_interval)
                    continue
                else:
                    # Error or unknown status
                    error_msg = result.get("error", f"Unknown status: {status}")
                    logger.error(f"Error processing request {request_id}: {error_msg}")
                    return {"status": "error", "error": error_msg}
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error retrieving result for request {request_id}: {str(e)}")
                if not wait:
                    raise
                time.sleep(poll_interval)
                continue
                
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.
        
        Args:
            request_id: ID of the request to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        if request_id not in self.pending_requests:
            logger.warning(f"Unknown request ID: {request_id}")
            return False
            
        try:
            response = self.session.post(
                f"{self.api_endpoint}/cancel/{request_id}",
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("cancelled", False):
                self.pending_requests[request_id]["status"] = "cancelled"
                logger.info(f"Successfully cancelled request {request_id}")
                return True
            else:
                logger.warning(f"Failed to cancel request {request_id}: {result.get('message', 'Unknown error')}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error cancelling request {request_id}: {str(e)}")
            return False
            
    def close(self):
        """Close the client session."""
        self.session.close()


# For simulation/testing when a real cloud backend is not available
class MockCloudClient:
    """
    Mock implementation of a cloud client for testing and development.
    
    This class simulates the behavior of a cloud API without making real network requests.
    """
    
    def __init__(self, latency: float = 0.2, failure_rate: float = 0.0):
        """
        Initialize the mock cloud client.
        
        Args:
            latency: Simulated processing time in seconds
            failure_rate: Probability of request failure (0.0 to 1.0)
        """
        self.latency = latency
        self.failure_rate = failure_rate
        self.requests = {}
        self.logger = logging.getLogger("mock_cloud")
        
    def send_tensor(self, tensor: np.ndarray, layer_idx: int, metadata: Dict[str, Any] = None) -> str:
        """Simulate sending a tensor to the cloud."""
        import random
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            self.logger.warning("Simulating random failure in send_tensor")
            raise requests.exceptions.RequestException("Simulated connection error")
            
        # Generate a mock request ID
        request_id = f"mock-{time.time()}-{random.randint(1000, 9999)}"
        
        # Store the request
        self.requests[request_id] = {
            "tensor": tensor,
            "layer_idx": layer_idx,
            "metadata": metadata or {},
            "status": "pending",
            "timestamp": time.time(),
            "result": None
        }
        
        self.logger.info(f"Mock cloud received tensor for layer {layer_idx}, request ID: {request_id}")
        return request_id
        
    def get_result(self, request_id: str, wait: bool = True, poll_interval: float = 0.5) -> Dict[str, Any]:
        """Simulate retrieving results from the cloud."""
        if request_id not in self.requests:
            self.logger.warning(f"Unknown request ID: {request_id}")
            return {"status": "error", "error": "Unknown request ID"}
            
        request = self.requests[request_id]
        
        # Check if enough time has passed to simulate processing
        elapsed = time.time() - request["timestamp"]
        
        if elapsed < self.latency:
            if not wait:
                return {"status": "processing"}
                
            # Wait until processing is complete
            time_to_wait = self.latency - elapsed
            self.logger.info(f"Mock cloud processing, waiting {time_to_wait:.2f}s for request {request_id}")
            time.sleep(time_to_wait)
        
        # Now the request is complete, generate a result
        if request["result"] is None:
            # In a real implementation, this would do actual computation
            # Here we just pass through or modify the tensor slightly
            input_tensor = request["tensor"]
            
            # Apply a simple transformation (e.g., add a small random noise)
            result_tensor = input_tensor + np.random.normal(0, 0.01, input_tensor.shape)
            
            request["result"] = {
                "tensor": result_tensor,
                "shape": result_tensor.shape,
                "dtype": str(result_tensor.dtype),
                "layer_idx": request["layer_idx"],
                "metadata": request["metadata"],
                "processing_time": self.latency
            }
            request["status"] = "completed"
            
        self.logger.info(f"Mock cloud returning result for request {request_id}")
        return {
            "status": "completed",
            **request["result"]
        }
        
    def cancel_request(self, request_id: str) -> bool:
        """Simulate cancelling a request."""
        if request_id not in self.requests:
            return False
            
        self.requests[request_id]["status"] = "cancelled"
        self.logger.info(f"Mock cloud cancelled request {request_id}")
        return True
        
    def close(self):
        """No-op for mock client."""
        pass 