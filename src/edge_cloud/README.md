# Edge-Cloud Collaborative Inference

This module implements edge-cloud collaborative inference for large language models, as described in the research on efficient mobile inference. The approach dynamically partitions computation between a mobile device and cloud servers to optimize for energy, latency, and memory usage.

## Key Components

### EdgeCloudManager

The central component that orchestrates all aspects of edge-cloud collaborative inference:

- Dynamically decides layer partitioning based on device conditions
- Routes tensor computations between local device and cloud
- Applies privacy protection for cloud-sent data
- Leverages mini-LLM for "easy" tokens

### DeviceMonitor

Tracks device conditions that influence partitioning decisions:

- Network bandwidth and latency
- CPU/GPU usage
- Memory availability

### CloudClient

Handles communication with cloud servers:

- Sends tensor data for remote computation
- Receives results from cloud processing
- Manages request tracking and error handling

### MiniLLM

Handles small on-device model for processing "easy" tokens:

- Assesses token difficulty
- Processes predictable tokens locally
- Reduces cloud dependency for simple predictions

### Security Module

Protects sensitive data during cloud communication:

- Encryption for transmitted tensors
- Dimensionality reduction for data privacy
- Secure channel management

## Usage Example

```python
# Set up the components
device_monitor = DeviceMonitor(background_monitoring=True)
cloud_client = CloudClient(api_key="your_api_key")
mini_llm = MiniLLMHandler(model=small_model, tokenizer=tokenizer)
privacy = PrivacyProtection(encryption_enabled=True)

# Create the Edge-Cloud Manager
edge_cloud_manager = EdgeCloudManager(
    model=main_model,
    device_monitor=device_monitor,
    cloud_client=cloud_client,
    mini_llm=mini_llm,
    privacy_protection=privacy,
    energy_weight=0.3,
    latency_weight=0.4,
    memory_weight=0.3
)

# Run inference with automatic partitioning
outputs = edge_cloud_manager.process_hybrid(input_ids)
```

## Configuration Options

The behavior can be customized with these parameters:

- **Weights**: Adjust importance of energy, latency, and memory in the cost function
- **Force modes**: Can force "local-only" or "cloud-only" execution for testing
- **Privacy settings**: Enable/disable encryption and dimensionality reduction
- **Mini-LLM threshold**: Control when the small model is used

## Running the Example

Use the provided shell script to run a complete example:

```
./src/run_edge_cloud.sh --tokens 20 --cloud-latency 0.5 --energy-weight 0.4
```

Options include:
- `--model`: Model name (default: distilgpt2)
- `--prompt`: Input prompt
- `--tokens`: Number of tokens to generate
- `--energy-weight`, `--latency-weight`, `--memory-weight`: Cost function weights
- `--cloud-latency`: Simulated cloud latency
- `--no-encryption`: Disable encryption
- `--force-local`, `--force-cloud`: Force execution mode 