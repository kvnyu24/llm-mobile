"""
Device Monitor for Edge-Cloud Collaborative Inference

This module provides functionality to monitor device conditions including
CPU/GPU usage, memory availability, network bandwidth, battery level, etc.
These metrics are used to make optimal partitioning decisions.
"""

import time
import threading
import subprocess
import logging
from typing import Dict, Any, Optional, Callable, List
import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("device_monitor")


class NetworkMonitor:
    """
    Monitors network conditions for making edge-cloud offloading decisions.
    
    Measures bandwidth, latency, and connectivity to determine when to
    offload computation to the cloud.
    """
    
    def __init__(self, test_endpoint: str = "https://www.google.com", 
                 history_length: int = 10):
        """
        Initialize the network monitor.
        
        Args:
            test_endpoint: URL to use for connectivity and latency tests
            history_length: Number of measurements to keep in history
        """
        self.test_endpoint = test_endpoint
        self.history_length = history_length
        self.bandwidth_history = []
        self.latency_history = []
        self.last_measurement_time = 0
        self.measurement_interval = 5  # seconds
        
    def measure_bandwidth(self) -> float:
        """
        Measure current network bandwidth.
        
        Returns:
            Estimated bandwidth in Mbps
        """
        # In a real implementation, this would perform an actual bandwidth test
        # For now, we'll use a simple approach of downloading a small file and measuring time
        try:
            start_time = time.time()
            response = subprocess.run(
                ["curl", "-s", "-w", "%{speed_download}", "-o", "/dev/null", self.test_endpoint],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Speed is in bytes/sec, convert to Mbps
            download_speed_bps = float(response.stdout.strip())
            bandwidth_mbps = download_speed_bps * 8 / 1_000_000
            
            # Add to history
            self.bandwidth_history.append(bandwidth_mbps)
            if len(self.bandwidth_history) > self.history_length:
                self.bandwidth_history.pop(0)
                
            return bandwidth_mbps
            
        except (subprocess.SubprocessError, ValueError) as e:
            logger.warning(f"Failed to measure bandwidth: {str(e)}")
            # Return last known value or a default
            return self.bandwidth_history[-1] if self.bandwidth_history else 1.0
            
    def measure_latency(self) -> float:
        """
        Measure current network latency.
        
        Returns:
            Latency in milliseconds
        """
        try:
            # Use ping to measure latency
            host = self.test_endpoint.replace("https://", "").replace("http://", "").split("/")[0]
            response = subprocess.run(
                ["ping", "-c", "3", host],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse the ping output to get average latency
            for line in response.stdout.splitlines():
                if "avg" in line:
                    # Format is typically like "min/avg/max/mdev = 12.345/23.456/34.567/5.678 ms"
                    avg_latency = float(line.split("=")[1].strip().split("/")[1])
                    
                    # Add to history
                    self.latency_history.append(avg_latency)
                    if len(self.latency_history) > self.history_length:
                        self.latency_history.pop(0)
                        
                    return avg_latency
            
            # If we couldn't parse the output, return the last known value or a default
            return self.latency_history[-1] if self.latency_history else 100.0
                
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            logger.warning(f"Failed to measure latency: {str(e)}")
            return self.latency_history[-1] if self.latency_history else 100.0
            
    def check_connectivity(self) -> bool:
        """
        Check if the device has internet connectivity.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = subprocess.run(
                ["ping", "-c", "1", "-W", "2", "8.8.8.8"],  # Google DNS
                capture_output=True,
                timeout=3
            )
            return response.returncode == 0
        except subprocess.SubprocessError:
            return False
            
    def get_network_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get the current network metrics.
        
        Args:
            force_refresh: Whether to force a new measurement
            
        Returns:
            Dictionary with network metrics
        """
        current_time = time.time()
        
        # Only measure if forced or enough time has passed since last measurement
        if force_refresh or current_time - self.last_measurement_time > self.measurement_interval:
            connected = self.check_connectivity()
            
            if connected:
                bandwidth = self.measure_bandwidth()
                latency = self.measure_latency()
            else:
                bandwidth = 0.0
                latency = float('inf')
                
            self.last_measurement_time = current_time
            
        else:
            # Use the most recent measurements
            connected = True  # Assume connected if we have recent measurements
            bandwidth = self.bandwidth_history[-1] if self.bandwidth_history else 1.0
            latency = self.latency_history[-1] if self.latency_history else 100.0
            
        # Calculate averages for stability
        avg_bandwidth = sum(self.bandwidth_history) / len(self.bandwidth_history) if self.bandwidth_history else bandwidth
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else latency
            
        return {
            "connected": connected,
            "bandwidth_mbps": bandwidth,
            "avg_bandwidth_mbps": avg_bandwidth,
            "latency_ms": latency,
            "avg_latency_ms": avg_latency,
            "timestamp": current_time
        }
        

class HardwareMonitor:
    """
    Monitors device hardware resources for making edge-cloud offloading decisions.
    
    Tracks CPU usage, GPU memory, RAM availability, and battery level.
    """
    
    def __init__(self, history_length: int = 10):
        """
        Initialize the hardware monitor.
        
        Args:
            history_length: Number of measurements to keep in history
        """
        self.history_length = history_length
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.gpu_memory_history = [] if TORCH_AVAILABLE else None
        self.battery_level_history = []
        self.last_measurement_time = 0
        self.measurement_interval = 2  # seconds
        
    def measure_cpu_usage(self) -> float:
        """
        Measure current CPU usage.
        
        Returns:
            CPU usage as a percentage (0-100)
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Add to history
        self.cpu_usage_history.append(cpu_percent)
        if len(self.cpu_usage_history) > self.history_length:
            self.cpu_usage_history.pop(0)
            
        return cpu_percent
        
    def measure_memory_usage(self) -> Dict[str, float]:
        """
        Measure current memory usage.
        
        Returns:
            Dictionary with memory metrics
        """
        memory = psutil.virtual_memory()
        metrics = {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_percent": memory.percent
        }
        
        # Add to history
        self.memory_usage_history.append(metrics)
        if len(self.memory_usage_history) > self.history_length:
            self.memory_usage_history.pop(0)
            
        return metrics
        
    def measure_gpu_memory(self) -> Optional[Dict[str, float]]:
        """
        Measure current GPU memory usage if available.
        
        Returns:
            Dictionary with GPU memory metrics, or None if GPU is not available
        """
        if not TORCH_AVAILABLE:
            return None
            
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                return None
                
            device = torch.device("cuda:0")
            gpu_props = torch.cuda.get_device_properties(device)
            
            # Get memory usage
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)    # MB
            total = gpu_props.total_memory / (1024 * 1024)                  # MB
            
            metrics = {
                "total_mb": total,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "available_mb": total - allocated,
                "used_percent": (allocated / total) * 100 if total > 0 else 0
            }
            
            # Add to history
            if self.gpu_memory_history is not None:
                self.gpu_memory_history.append(metrics)
                if len(self.gpu_memory_history) > self.history_length:
                    self.gpu_memory_history.pop(0)
                    
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to measure GPU memory: {str(e)}")
            return None
            
    def measure_battery_level(self) -> Optional[float]:
        """
        Measure current battery level if available.
        
        Returns:
            Battery percentage (0-100) or None if not applicable
        """
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return None
                
            level = battery.percent
            
            # Add to history
            self.battery_level_history.append(level)
            if len(self.battery_level_history) > self.history_length:
                self.battery_level_history.pop(0)
                
            return level
            
        except Exception:
            return None
            
    def get_hardware_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get the current hardware metrics.
        
        Args:
            force_refresh: Whether to force a new measurement
            
        Returns:
            Dictionary with hardware metrics
        """
        current_time = time.time()
        
        # Only measure if forced or enough time has passed since last measurement
        if force_refresh or current_time - self.last_measurement_time > self.measurement_interval:
            cpu_usage = self.measure_cpu_usage()
            memory_usage = self.measure_memory_usage()
            gpu_memory = self.measure_gpu_memory()
            battery_level = self.measure_battery_level()
            
            self.last_measurement_time = current_time
            
        else:
            # Use the most recent measurements
            cpu_usage = self.cpu_usage_history[-1] if self.cpu_usage_history else 0.0
            memory_usage = self.memory_usage_history[-1] if self.memory_usage_history else {"total_mb": 0, "available_mb": 0, "used_percent": 0}
            gpu_memory = self.gpu_memory_history[-1] if self.gpu_memory_history else None
            battery_level = self.battery_level_history[-1] if self.battery_level_history else None
            
        # Calculate averages for stability
        avg_cpu_usage = sum(self.cpu_usage_history) / len(self.cpu_usage_history) if self.cpu_usage_history else cpu_usage
            
        metrics = {
            "cpu_usage_percent": cpu_usage,
            "avg_cpu_usage_percent": avg_cpu_usage,
            "memory": memory_usage,
            "battery_level_percent": battery_level,
            "timestamp": current_time
        }
        
        if gpu_memory is not None:
            metrics["gpu_memory"] = gpu_memory
            
        return metrics
        

class DeviceMonitor:
    """
    Combined monitor for all device conditions relevant to edge-cloud decisions.
    
    Provides a unified interface to network and hardware metrics.
    """
    
    def __init__(self, 
                 network_test_endpoint: str = "https://www.google.com",
                 monitoring_interval: float = 5.0,
                 history_length: int = 10,
                 background_monitoring: bool = False,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the device monitor.
        
        Args:
            network_test_endpoint: URL to use for connectivity and latency tests
            monitoring_interval: How often to refresh metrics in background mode (seconds)
            history_length: Number of measurements to keep in history
            background_monitoring: Whether to continuously monitor in the background
            callback: Function to call with new metrics in background mode
        """
        self.network_monitor = NetworkMonitor(
            test_endpoint=network_test_endpoint,
            history_length=history_length
        )
        
        self.hardware_monitor = HardwareMonitor(
            history_length=history_length
        )
        
        self.monitoring_interval = monitoring_interval
        self.background_monitoring = background_monitoring
        self.callback = callback
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        
        # Start background monitoring if requested
        if background_monitoring:
            self.start_background_monitoring()
            
    def get_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get combined metrics from all monitors.
        
        Args:
            force_refresh: Whether to force a new measurement
            
        Returns:
            Dictionary with all metrics
        """
        network_metrics = self.network_monitor.get_network_metrics(force_refresh)
        hardware_metrics = self.hardware_monitor.get_hardware_metrics(force_refresh)
        
        return {
            "network": network_metrics,
            "hardware": hardware_metrics,
            "timestamp": time.time()
        }
        
    def start_background_monitoring(self):
        """Start continuous monitoring in a background thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
            
        self.stop_flag.clear()
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_background_monitoring(self):
        """Stop the background monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_flag.set()
            self.monitor_thread.join(timeout=2.0)
            
    def is_monitoring(self):
        """Check if background monitoring is currently active.
        
        Returns:
            bool: True if monitoring thread is running, False otherwise
        """
        return self.monitor_thread is not None and self.monitor_thread.is_alive()
            
    def _background_monitor(self):
        """Background thread that continuously collects metrics."""
        while not self.stop_flag.is_set():
            try:
                metrics = self.get_metrics(force_refresh=True)
                
                if self.callback:
                    self.callback(metrics)
                    
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
                
            # Sleep for the monitoring interval
            self.stop_flag.wait(self.monitoring_interval)
            
    def __del__(self):
        """Ensure the background thread is stopped when the object is deleted."""
        self.stop_background_monitoring() 