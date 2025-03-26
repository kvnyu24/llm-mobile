"""
Security Utilities for Edge-Cloud Communication

This module provides encryption, compression, and other security features
for protecting data exchanged with cloud services.
"""

import os
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


class Encryptor:
    """
    Handles encryption of model states before sending to the cloud.
    
    Uses AES encryption with CBC mode for secure communication.
    """
    
    def __init__(self, key: Optional[bytes] = None, iv: Optional[bytes] = None):
        """
        Initialize the encryptor.
        
        Args:
            key: Encryption key (32 bytes for AES-256)
            iv: Initialization vector (16 bytes)
        """
        # Generate a key if none provided
        self.key = key or os.urandom(32)  # 256-bit key
        self.iv = iv or os.urandom(16)    # 128-bit IV for AES
        self.backend = default_backend()
        
    def encrypt_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        """
        Encrypt a NumPy tensor.
        
        Args:
            tensor: NumPy array to encrypt
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        # Convert tensor to bytes
        tensor_bytes = tensor.tobytes()
        
        # Pad the data to a multiple of block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(tensor_bytes) + padder.finalize()
        
        # Encrypt the padded data
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=self.backend)
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return the encrypted data along with metadata needed for decryption
        return {
            "encrypted_data": encrypted_data,
            "iv": self.iv,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype)
        }
        
    def decrypt_tensor(self, encrypted_package: Dict[str, Any]) -> np.ndarray:
        """
        Decrypt an encrypted tensor.
        
        Args:
            encrypted_package: Dictionary with encrypted data and metadata
            
        Returns:
            Decrypted NumPy array
        """
        encrypted_data = encrypted_package["encrypted_data"]
        iv = encrypted_package.get("iv", self.iv)
        shape = encrypted_package["shape"]
        dtype = encrypted_package["dtype"]
        
        # Decrypt the data
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Unpad the data
        unpadder = padding.PKCS7(128).unpadder()
        unpadded = unpadder.update(decrypted_padded) + unpadder.finalize()
        
        # Convert back to a NumPy array
        tensor = np.frombuffer(unpadded, dtype=dtype).reshape(shape)
        return tensor
        

class DimensionalityReducer:
    """
    Compresses tensors by reducing their dimensionality.
    
    This both improves communication efficiency and enhances privacy by
    not sending the full representation to the cloud.
    """
    
    def __init__(self, reduction_method: str = "pca", target_dims: Optional[int] = None, 
                 reduction_ratio: float = 0.5):
        """
        Initialize the dimensionality reducer.
        
        Args:
            reduction_method: Method to use ("pca" or "random_projection")
            target_dims: Target number of dimensions (if None, use reduction_ratio)
            reduction_ratio: Fraction of original dimensions to keep (if target_dims is None)
        """
        self.reduction_method = reduction_method
        self.target_dims = target_dims
        self.reduction_ratio = reduction_ratio
        self.fitted = False
        self.reducer = None
        
    def fit(self, sample_data: np.ndarray):
        """
        Fit the reducer using sample data.
        
        Args:
            sample_data: Data to fit the dimensionality reduction model
        """
        original_shape = sample_data.shape
        
        # Reshape to 2D if needed
        if len(original_shape) > 2:
            sample_data = sample_data.reshape(original_shape[0], -1)
            
        # Determine number of components
        if self.target_dims is None:
            n_components = max(1, int(sample_data.shape[1] * self.reduction_ratio))
        else:
            n_components = min(self.target_dims, sample_data.shape[1])
            
        # Create and fit the reducer
        if self.reduction_method == "pca":
            self.reducer = PCA(n_components=n_components)
        else:  # random projection
            self.reducer = GaussianRandomProjection(n_components=n_components)
            
        self.reducer.fit(sample_data)
        self.original_shape = original_shape
        self.fitted = True
        
    def reduce(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduce the dimensionality of a tensor.
        
        Args:
            tensor: The tensor to compress
            
        Returns:
            Tuple of (reduced_tensor, metadata_for_reconstruction)
        """
        if not self.fitted:
            self.fit(tensor)
            
        original_shape = tensor.shape
        
        # Reshape to 2D if needed
        if len(original_shape) > 2:
            tensor_2d = tensor.reshape(original_shape[0], -1)
        else:
            tensor_2d = tensor
            
        # Apply dimensionality reduction
        reduced = self.reducer.transform(tensor_2d)
        
        # Store metadata needed for approximate reconstruction
        metadata = {
            "original_shape": original_shape,
            "reduction_method": self.reduction_method,
            "n_components": reduced.shape[1]
        }
        
        # For PCA, we also store components for reconstruction
        if self.reduction_method == "pca":
            metadata["components"] = self.reducer.components_
            metadata["mean"] = self.reducer.mean_
            
        return reduced, metadata
        
    def reconstruct(self, reduced_tensor: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Approximately reconstruct the original tensor from reduced form.
        
        Args:
            reduced_tensor: The reduced tensor
            metadata: Metadata from the reduction process
            
        Returns:
            Reconstructed tensor (approximate)
        """
        original_shape = metadata["original_shape"]
        
        # For PCA, we can do actual reconstruction
        if metadata["reduction_method"] == "pca":
            components = metadata["components"]
            mean = metadata["mean"]
            
            # Reconstruct using the PCA components
            reconstructed_2d = np.dot(reduced_tensor, components) + mean
        else:
            # For random projection, best we can do is a placeholder with zeros
            # In a real implementation, you might use more sophisticated methods
            flat_size = np.prod(original_shape[1:])
            reconstructed_2d = np.zeros((original_shape[0], flat_size))
            
        # Reshape back to original shape
        reconstructed = reconstructed_2d.reshape(original_shape)
        return reconstructed


class PrivacyProtection:
    """
    Comprehensive privacy protection for cloud communication.
    
    Combines encryption and dimensionality reduction to protect data
    sent to the cloud.
    """
    
    def __init__(self, 
                 encryption_enabled: bool = True, 
                 reduction_enabled: bool = True,
                 encryption_key: Optional[bytes] = None,
                 reduction_ratio: float = 0.5):
        """
        Initialize privacy protection.
        
        Args:
            encryption_enabled: Whether to enable encryption
            reduction_enabled: Whether to enable dimensionality reduction
            encryption_key: Key for encryption (generated if None)
            reduction_ratio: How much to reduce dimensions by
        """
        self.encryption_enabled = encryption_enabled
        self.reduction_enabled = reduction_enabled
        
        # Initialize components
        if encryption_enabled:
            self.encryptor = Encryptor(key=encryption_key)
        
        if reduction_enabled:
            self.reducer = DimensionalityReducer(reduction_ratio=reduction_ratio)
            
    def protect(self, tensor: np.ndarray) -> Dict[str, Any]:
        """
        Apply privacy protection to a tensor.
        
        Args:
            tensor: Tensor to protect
            
        Returns:
            Protected data package
        """
        protected_data = {
            "original_shape": tensor.shape,
            "original_dtype": str(tensor.dtype)
        }
        
        current_tensor = tensor
        
        # First apply dimensionality reduction if enabled
        if self.reduction_enabled:
            reduced_tensor, reduction_metadata = self.reducer.reduce(current_tensor)
            protected_data["reduction_metadata"] = reduction_metadata
            current_tensor = reduced_tensor
            
        # Then apply encryption if enabled
        if self.encryption_enabled:
            encrypted_package = self.encryptor.encrypt_tensor(current_tensor)
            protected_data["encrypted_package"] = encrypted_package
            # The actual data is now in the encrypted package
        else:
            # If not encrypted, store the tensor directly
            protected_data["tensor"] = current_tensor
            
        return protected_data
        
    def unprotect(self, protected_data: Dict[str, Any]) -> np.ndarray:
        """
        Recover the original tensor from protected data.
        
        Args:
            protected_data: Protected data package
            
        Returns:
            Reconstructed tensor (may be approximate if reduction was used)
        """
        # First decrypt if encryption was used
        if self.encryption_enabled:
            encrypted_package = protected_data["encrypted_package"]
            current_tensor = self.encryptor.decrypt_tensor(encrypted_package)
        else:
            current_tensor = protected_data["tensor"]
            
        # Then reconstruct if dimensionality reduction was used
        if self.reduction_enabled:
            reduction_metadata = protected_data["reduction_metadata"]
            current_tensor = self.reducer.reconstruct(current_tensor, reduction_metadata)
            
        return current_tensor 