import torch
from safetensors.torch import load_file, save_file
from typing import Dict
import argparse
import os

# Configs
N_LAYERS = 61
N_HEADS = 128
Q_LORA_RANK = 1536
V_HEAD_DIM = 128
GATE_DIM = N_HEADS * V_HEAD_DIM  # 128 * 128 = 16384
BLOCK_SIZE = 128  # FP8 quantization block size


def create_gate_weights(output_path: str, fp8: bool = False) -> None:
    """
    Create gate weights file for all layers.
    Initializes to zeros so sigmoid(0) = 0.5 (neutral gating).
    
    Args:
        output_path: Where to save the safetensors file
        fp8: If True, create FP8 weights with scales. If False, create BF16 weights.
    """
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    
    print(f"Creating gate weights for {N_LAYERS} layers...")
    print(f"  Gate dim: {GATE_DIM} (n_heads={N_HEADS} × v_head_dim={V_HEAD_DIM})")
    print(f"  Input dim: {Q_LORA_RANK}")
    print(f"  Format: {'FP8 + scales' if fp8 else 'BF16'}")
    
    gate_weights = {}
    
    for layer_idx in range(N_LAYERS):
        weight_key = f"model.layers.{layer_idx}.self_attn.q_b_proj.gate_weight"
        
        # Create zero-initialized weights
        gate_weights[weight_key] = torch.zeros(GATE_DIM, Q_LORA_RANK, dtype=dtype)
        
        # For FP8, also create scale tensor
        if fp8:
            scale_key = f"model.layers.{layer_idx}.self_attn.q_b_proj.gate_weight_scale"
            scale_out = (GATE_DIM + BLOCK_SIZE - 1) // BLOCK_SIZE
            scale_in = (Q_LORA_RANK + BLOCK_SIZE - 1) // BLOCK_SIZE
            # Initialize scales to 1.0 (neutral scaling for zero weights)
            gate_weights[scale_key] = torch.ones(scale_out, scale_in, dtype=torch.float32)
        
        if (layer_idx + 1) % 10 == 0:
            print(f"  Created {layer_idx + 1}/{N_LAYERS} layers")
    
    print(f"\nSaving to {output_path}...")
    save_file(gate_weights, output_path)
    
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Done! File size: {size_gb:.2f} GB")


class GateWeightLoader:
    """
    Helper class to load and combine weights during model initialization.
    Handles both BF16 and FP8 quantized weights.
    
    Usage:
        loader = GateWeightLoader("gate_weights.safetensors")
        
        # When loading each shard:
        weights = load_file(shard_path)
        weights = loader.apply(weights)
    """
    
    def __init__(self, gate_weights_path: str):
        self.gate_weights = load_file(gate_weights_path)
        print(f"Loaded gate weights from {gate_weights_path}")
    
    def apply(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand any q_b_proj weights found in this shard."""
        for key in list(weights.keys()):
            # Handle weight tensors
            if "q_b_proj.weight" in key and "scale" not in key:
                gate_key = key.replace(".weight", ".gate_weight")
                if gate_key in self.gate_weights:
                    weights[key] = torch.cat([
                        weights[key],
                        self.gate_weights[gate_key].to(weights[key].dtype)
                    ], dim=0)
            
            # Handle FP8 scale tensors
            elif "q_b_proj.weight_scale" in key:
                gate_scale_key = key.replace(".weight_scale", ".gate_weight_scale")
                if gate_scale_key in self.gate_weights:
                    weights[key] = torch.cat([
                        weights[key],
                        self.gate_weights[gate_scale_key]
                    ], dim=0)
        
        return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Gate Weights Manager")
    parser.add_argument("--create", action="store_true", help="Create gate weights file")
    parser.add_argument("--output", "-o", type=str, default="gate_weights.safetensors")
    parser.add_argument("--fp8", action="store_true", help="Create FP8 quantized weights with scales")
    
    args = parser.parse_args()
    
    if args.create:
        create_gate_weights(args.output, fp8=args.fp8)
    else:
        print("Use --create to generate gate weights file")
        print("Examples:")
        print("  BF16: python gate_weights.py --create --output gate_weights.safetensors")
        print("  FP8:  python gate_weights.py --create --output gate_weights.safetensors --fp8")
