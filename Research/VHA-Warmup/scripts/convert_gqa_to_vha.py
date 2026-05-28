#!/usr/bin/env python3
"""
Convert qwen3_GQA_1p7B_pretrain checkpoint (16Q-8KV) to VHA format (8Q-2KV + premix).

Source: /root/paddlejob/share-storage/gpfs/system-public/dingxibo/ckpts/qwen3_GQA_1p7B_pretrain/
Target: VHA checkpoint with premix_weight, postmix_U/V

Usage:
    python convert_gqa_to_vha.py \
        --source_path /root/paddlejob/share-storage/gpfs/system-public/dingxibo/ckpts/qwen3_GQA_1p7B_pretrain \
        --output_path /root/paddlejob/share-storage/gpfs/system-public/dingxibo/ckpts/qwen3_VHA_1p7B_from_GQA
"""

import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vha_warmup.initializer import VHAInitializer


def load_safetensors(path):
    """Load all tensors from a safetensors file as float32 numpy arrays.

    Handles bfloat16 stored as uint16 by using ml_dtypes to properly decode.
    """
    import ml_dtypes
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)
            # safetensors stores bfloat16 as uint16 in numpy mode
            if arr.dtype == np.uint16:
                arr = arr.view(ml_dtypes.bfloat16).astype(np.float32)
            elif arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            tensors[key] = arr
    return tensors


def to_bf16_bytes(arr):
    """Convert numpy float32 array to bfloat16 bytes via uint16 view."""
    # numpy doesn't natively support bfloat16, so we use the ml_dtypes package
    # or manually truncate float32 -> bfloat16
    import ml_dtypes
    return arr.astype(ml_dtypes.bfloat16)


def main():
    parser = argparse.ArgumentParser(description="Convert GQA checkpoint to VHA format")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--source_q_heads", type=int, default=16)
    parser.add_argument("--source_kv_heads", type=int, default=8)
    parser.add_argument("--target_q_heads", type=int, default=8)
    parser.add_argument("--target_kv_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--postmix_rank", type=int, default=4)
    parser.add_argument("--kv_merge_method", type=str, default="average",
                        choices=["average", "svd"],
                        help="KV merge method: 'average' (simple mean) or 'svd' (Procrustes align)")
    parser.add_argument("--joint", action="store_true", default=True,
                        help="Use joint Q decomposition with Procrustes-blended W2 (1000x better cond num)")
    parser.add_argument("--no_joint", dest="joint", action="store_false",
                        help="Disable joint decomposition, use standard least-squares W2")
    args = parser.parse_args()

    initializer = VHAInitializer(
        source_num_q_heads=args.source_q_heads,
        source_num_kv_heads=args.source_kv_heads,
        target_num_q_heads=args.target_q_heads,
        target_num_kv_heads=args.target_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
    )

    os.makedirs(args.output_path, exist_ok=True)

    # Load source checkpoint
    source_file = os.path.join(args.source_path, "model-00001-of-00001.safetensors")
    print(f"Loading source checkpoint: {source_file}")
    source_tensors = load_safetensors(source_file)
    print(f"  Loaded {len(source_tensors)} tensors")

    # Check source dtype
    sample_key = "model.layers.0.self_attn.q_proj.weight"
    src_dtype = source_tensors[sample_key].dtype
    print(f"  Source dtype: {src_dtype}")

    import ml_dtypes  # for bfloat16 support

    print(f"\nConverting {args.num_layers} layers...")
    print(f"  Source: {args.source_q_heads}Q-{args.source_kv_heads}KV")
    print(f"  Target: {args.target_q_heads}Q-{args.target_kv_heads}KV + premix")
    print(f"  Virtual heads: {args.target_q_heads * args.target_kv_heads}")
    print(f"  KV merge method: {args.kv_merge_method}")
    print(f"  Joint Q-KV decomposition: {args.joint}")

    output_tensors = {}
    total_errors = []
    total_heads = args.target_q_heads * args.target_kv_heads

    for layer_idx in range(args.num_layers):
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Load and cast to float32 for computation
        W_q = source_tensors[f"{prefix}.q_proj.weight"].astype(np.float32)
        W_k = source_tensors[f"{prefix}.k_proj.weight"].astype(np.float32)
        W_v = source_tensors[f"{prefix}.v_proj.weight"].astype(np.float32)
        W_o = source_tensors[f"{prefix}.o_proj.weight"].astype(np.float32)

        # Run VHA conversion
        result = initializer.convert_layer(W_q, W_k, W_v, W_o,
                                           kv_merge_method=args.kv_merge_method,
                                           joint=args.joint)

        # Compute approximation error
        q_result = initializer.decompose_q_weights(W_q)
        errors = initializer.compute_approximation_error(W_q, q_result)
        total_errors.append(errors)

        # Store converted attention weights as fused qkv_proj (cast back to bfloat16)
        # qkv_proj layout: [hidden_size, (tgt_H_q * d + tgt_H_k * d + tgt_H_k * d)]
        # = [2048, (8*128 + 2*128 + 2*128)] = [2048, 1536]
        q_bf16 = result["q_proj"].astype(ml_dtypes.bfloat16)
        k_bf16 = result["k_proj"].astype(ml_dtypes.bfloat16)
        v_bf16 = result["v_proj"].astype(ml_dtypes.bfloat16)
        qkv_fused = np.concatenate([q_bf16, k_bf16, v_bf16], axis=1)
        output_tensors[f"{prefix}.qkv_proj.weight"] = qkv_fused
        output_tensors[f"{prefix}.o_proj.weight"] = result["o_proj"].astype(ml_dtypes.bfloat16)

        # VHA premix weight: w2_rot [tgt_H_k, d, d] -> vha_premix_weight
        output_tensors[f"{prefix}.vha_premix_weight"] = result["w2_rot"].astype(ml_dtypes.bfloat16)

        # VHA postmix: zero/small init
        postmix_U = np.random.randn(total_heads, args.postmix_rank).astype(np.float32) * 0.01
        postmix_V = np.zeros((total_heads, args.postmix_rank), dtype=np.float32)
        output_tensors[f"{prefix}.vha_postmix_U"] = postmix_U.astype(ml_dtypes.bfloat16)
        output_tensors[f"{prefix}.vha_postmix_V"] = postmix_V.astype(ml_dtypes.bfloat16)

        # Copy q_norm, k_norm directly
        output_tensors[f"{prefix}.q_norm.weight"] = source_tensors[f"{prefix}.q_norm.weight"]
        output_tensors[f"{prefix}.k_norm.weight"] = source_tensors[f"{prefix}.k_norm.weight"]

        # Copy non-attention weights directly (fuse gate_proj + up_proj -> up_gate_proj)
        layer_prefix = f"model.layers.{layer_idx}"
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.down_proj.weight",
        ]:
            key = f"{layer_prefix}.{suffix}"
            output_tensors[key] = source_tensors[key]

        # Fuse gate_proj and up_proj into up_gate_proj: [hidden_size, 2 * intermediate_size]
        gate_proj = source_tensors[f"{layer_prefix}.mlp.gate_proj.weight"]
        up_proj = source_tensors[f"{layer_prefix}.mlp.up_proj.weight"]
        up_gate_fused = np.concatenate([up_proj, gate_proj], axis=1)
        output_tensors[f"{layer_prefix}.mlp.up_gate_proj.weight"] = up_gate_fused

        print(f"  Layer {layer_idx:2d}: Q errors = {errors}")

    # Copy embedding and final norm
    # Source uses "model.embed_tokens.weight", target expects "model.embedding.embed_tokens.weight"
    output_tensors["model.embedding.embed_tokens.weight"] = source_tensors["model.embed_tokens.weight"]
    output_tensors["model.norm.weight"] = source_tensors["model.norm.weight"]
    output_tensors["model.lm_head.weight"] = source_tensors["lm_head.weight"]

    # Summary
    print("\n=== Conversion Summary ===")
    for g in range(args.target_kv_heads):
        errs = [e[f"group_{g}"] for e in total_errors]
        print(f"  Group {g} avg relative error: {np.mean(errs):.6f}")

    # Save output checkpoint
    output_file = os.path.join(args.output_path, "model-00001-of-00001.safetensors")
    print(f"\nSaving VHA checkpoint to: {output_file}")
    save_file(output_tensors, output_file)
    print(f"  Saved {len(output_tensors)} tensors")

    # Generate weight map index
    weight_map = {k: "model-00001-of-00001.safetensors" for k in output_tensors.keys()}
    total_size = sum(t.nbytes for t in output_tensors.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_file = os.path.join(args.output_path, "model.safetensors.index.json")
    with open(index_file, "w") as f:
        json.dump(index, f, indent=4)

    # Generate VHA config.json
    source_config_path = os.path.join(args.source_path, "config.json")
    with open(source_config_path, "r") as f:
        config = json.load(f)

    # Update config for VHA
    config["attn_type"] = "vha"
    config["num_attention_heads"] = args.target_q_heads
    config["num_key_value_heads"] = args.target_kv_heads
    config["vha_enable_premix"] = True
    config["vha_enable_postmix"] = True
    config["vha_postmix_rank"] = args.postmix_rank

    config_file = os.path.join(args.output_path, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
