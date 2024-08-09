import os
import numpy as np
from safetensors import safe_open
from safetensors.tensorflow import save_file as safe_save  # Import the save function

def shard_safetensors(input_path, output_prefix, num_shards):
    """Shards a SafeTensors file into multiple files.

    Args:
        input_path (str): Path to the input SafeTensors file.
        output_prefix (str): Prefix for the output shard file names.
        num_shards (int): Number of shards to create.
    """

    with safe_open(input_path, framework="numpy") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    total_size = sum(tensor.nbytes for tensor in tensors.values())
    shard_size = total_size // num_shards

    shard_files = []
    current_shard_size = 0
    current_shard = {}
    shard_index = 0

    for name, tensor in tensors.items():
        if current_shard_size + tensor.nbytes > shard_size and shard_index < num_shards - 1:
            shard_files.append(current_shard)
            current_shard = {}
            current_shard_size = 0
            shard_index += 1

        current_shard[name] = tensor
        current_shard_size += tensor.nbytes

    shard_files.append(current_shard)  # Add the last shard

    for i, shard in enumerate(shard_files):
        shard_path = f"{output_prefix}-{i:04d}-of-{num_shards:04d}.safetensors"
        safe_save(shard, shard_path)  # Save using the correct function

# Example usage:
shard_safetensors('model.safetensors', 'model', 5)