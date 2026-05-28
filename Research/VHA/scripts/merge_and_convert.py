#!/usr/bin/env python3
"""
Merge PaddleFleet distributed checkpoint (.distcp) and convert to HuggingFace format.

Handles the full pipeline:
  Step 1: Merge .distcp shards into safetensors (using Paddle's merge_sharded_state_dict)
  Step 2: Convert Paddle safetensors to HF format (transpose + split fused weights)

Key differences between PaddleFleet and HF weight formats:
  - Paddle Linear: weight shape [in_features, out_features]
  - HF Linear:     weight shape [out_features, in_features]
  - Paddle fuses qkv_proj in GROUPED layout: [G0_Q, G0_K, G0_V, G1_Q, G1_K, G1_V, ...]
  - Paddle fuses up_gate_proj as: [gate, up] (first half = gate through silu, second = up)

Usage:
    # GQA 4B:
    python scripts/merge_and_convert.py \
        --input ./output/qwen3_gqa_4B_pretrain/checkpoint-24000/model_state \
        --output ./ckpts/qwen3_gqa_4B_hf \
        --config ./config/qwen3/Qwen3-4B-GQA/config.json \
        --tokenizer ./config/qwen3/tokenizer

    # VHA 4B:
    python scripts/merge_and_convert.py \
        --input ./output/qwen3_vha_4B_pretrain/checkpoint-24000/model_state \
        --output ./ckpts/qwen3_vha_4B_hf \
        --config ./config/qwen3/Qwen3-4B-VHA/config.json \
        --tokenizer ./config/qwen3/tokenizer

    # If already merged (skip Step 1):
    python scripts/merge_and_convert.py \
        --input ./output/qwen3_gqa_4B_pretrain/checkpoint-24000/model_state_merged \
        --output ./ckpts/qwen3_gqa_4B_hf \
        --config ./config/qwen3/Qwen3-4B-GQA/config.json \
        --tokenizer ./config/qwen3/tokenizer
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# ============================================================================
# Step 1: Merge distcp shards into safetensors
# ============================================================================

def needs_merge(input_dir: str) -> bool:
    """Check if input contains .distcp files (needs merging) or already has safetensors."""
    input_path = Path(input_dir)
    has_safetensors = list(input_path.glob("model-*.safetensors"))
    has_distcp = list(input_path.glob("*.distcp"))
    if has_safetensors:
        return False
    if has_distcp:
        return True
    raise FileNotFoundError(
        f"No .distcp or model-*.safetensors files found in {input_dir}"
    )


def merge_distcp(input_dir: str, merged_dir: str) -> None:
    """Merge distributed checkpoint shards into safetensors using Paddle."""
    print("=" * 60)
    print("  Step 1: Merging distcp shards -> safetensors")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {merged_dir}")
    print("=" * 60)

    import paddle
    from paddle.distributed.flex_checkpoint.dcp.load_state_dict import merge_sharded_state_dict

    os.makedirs(merged_dir, exist_ok=True)

    merge_sharded_state_dict(
        load_path=input_dir,
        save_path=merged_dir,
        prefix="model",
        safetensor_prefix="model",
        offload=True,
    )

    # Verify output
    merged_files = sorted(Path(merged_dir).glob("model-*.safetensors"))
    if not merged_files:
        raise RuntimeError(f"Merge failed: no safetensors files in {merged_dir}")
    print(f"\n  Merged into {len(merged_files)} safetensors file(s)")
    for f in merged_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name}: {size_mb:.1f} MB")
    print()


# ============================================================================
# Step 2: Convert Paddle safetensors to HF format
# ============================================================================

# Linear weight keys that need transposing (Paddle [in, out] -> HF [out, in])
TRANSPOSE_WEIGHT_KEYS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "up_gate_proj",
]


def should_transpose(key: str) -> bool:
    """Check if a weight key should be transposed."""
    for trans_key in TRANSPOSE_WEIGHT_KEYS:
        if f".{trans_key}.weight" in key or key == f"{trans_key}.weight":
            return True
    return False


def fix_dtype(tensor):
    """Reinterpret uint16 as bfloat16 (PaddleFleet may save bf16 as uint16 in safetensors)."""
    import torch
    if tensor.dtype == torch.uint16:
        return tensor.view(torch.bfloat16)
    return tensor


def split_qkv(tensor, num_heads: int, num_kv_heads: int, head_dim: int):
    """Split fused qkv_proj weight into separate q/k/v projections.

    PaddleFleet stores QKV in GROUPED layout (not simple [Q;K;V] concatenation):
      [G0_Q, G0_K, G0_V, G1_Q, G1_K, G1_V, ...]
    where each group has heads_per_group Q heads + 1 K head + 1 V head.

    After transpose, tensor shape is:
      [num_kv_heads * (heads_per_group + 2) * head_dim, hidden_size]
    """
    import torch
    heads_per_group = num_heads // num_kv_heads
    group_dim = (heads_per_group + 2) * head_dim
    hidden_size = tensor.shape[1]

    # Reshape to [num_kv_heads, group_dim, hidden_size]
    grouped = tensor.reshape(num_kv_heads, group_dim, hidden_size)

    q_dim_per_group = heads_per_group * head_dim
    q_parts = grouped[:, :q_dim_per_group, :]
    k_parts = grouped[:, q_dim_per_group:q_dim_per_group + head_dim, :]
    v_parts = grouped[:, q_dim_per_group + head_dim:, :]

    # Flatten: [num_heads * head_dim, hidden_size], [num_kv_heads * head_dim, hidden_size]
    q = q_parts.reshape(num_heads * head_dim, hidden_size)
    k = k_parts.reshape(num_kv_heads * head_dim, hidden_size)
    v = v_parts.reshape(num_kv_heads * head_dim, hidden_size)
    return q, k, v


def split_up_gate(tensor, intermediate_size: int):
    """Split fused up_gate_proj weight into separate gate/up projections.

    PaddleFleet layout: [gate; up] along output dimension.
    - First half (gate): passed through silu activation
    - Second half (up): multiplied with gate output
    """
    gate, up = tensor.split([intermediate_size, intermediate_size], dim=0)
    return gate, up


def convert_to_hf(merged_dir: str, output_dir: str, config_path: str, tokenizer_dir: str) -> None:
    """Convert merged Paddle safetensors to HF-compatible format."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    print("=" * 60)
    print("  Step 2: Converting Paddle safetensors -> HF format")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Load config for split dimensions
    with open(config_path) as f_cfg:
        config = json.load(f_cfg)
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim", config["hidden_size"] // num_heads)
    intermediate_size = config["intermediate_size"]
    tie_word_embeddings = config.get("tie_word_embeddings", False)
    print(f"  Config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}, intermediate={intermediate_size}, "
          f"tie_word_embeddings={tie_word_embeddings}")

    merged_path = Path(merged_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(merged_path.glob("model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No model-*.safetensors files found in {merged_dir}")

    print(f"  Found {len(shard_files)} shard(s)")

    weight_map = {}
    total_size = 0

    for shard_file in shard_files:
        shard_name = shard_file.name
        print(f"  Processing {shard_name}...")

        f = safe_open(str(shard_file), framework="pt")
        new_tensors = {}
        n_reinterpreted = 0
        n_transposed = 0
        n_split = 0

        for key in f.keys():
            tensor = f.get_tensor(key)

            # Fix dtype (uint16 -> bfloat16)
            original_dtype = tensor.dtype
            tensor = fix_dtype(tensor)
            if original_dtype == torch.uint16:
                n_reinterpreted += 1

            # Transpose linear weights: Paddle [in, out] -> HF [out, in]
            if tensor.ndim == 2 and should_transpose(key):
                tensor = tensor.T.contiguous()
                n_transposed += 1

            # Map Paddle key names to HF key names
            if key == "embedding.embed_tokens.weight":
                hf_key = "model.embed_tokens.weight"
            elif key == "norm.weight":
                hf_key = "model.norm.weight"
            elif key == "lm_head.weight":
                hf_key = "lm_head.weight"
            elif key.startswith("layers."):
                hf_key = "model." + key
            else:
                hf_key = "model." + key

            # Split fused weights
            if ".qkv_proj.weight" in key:
                q, k, v = split_qkv(tensor, num_heads, num_kv_heads, head_dim)
                base = hf_key.replace("qkv_proj.weight", "")
                new_tensors[base + "q_proj.weight"] = q
                new_tensors[base + "k_proj.weight"] = k
                new_tensors[base + "v_proj.weight"] = v
                n_split += 1
            elif ".up_gate_proj.weight" in key:
                gate, up = split_up_gate(tensor, intermediate_size)
                base = hf_key.replace("up_gate_proj.weight", "")
                new_tensors[base + "gate_proj.weight"] = gate
                new_tensors[base + "up_proj.weight"] = up
                n_split += 1
            else:
                new_tensors[hf_key] = tensor

        # For tied embeddings, ensure lm_head.weight exists
        if tie_word_embeddings and "lm_head.weight" not in new_tensors:
            if "model.embed_tokens.weight" in new_tensors:
                new_tensors["lm_head.weight"] = new_tensors["model.embed_tokens.weight"]

        # Update weight_map and save
        for k, t in new_tensors.items():
            weight_map[k] = shard_name
            total_size += t.numel() * t.element_size()

        save_file(new_tensors, str(output_path / shard_name))
        print(f"    {len(new_tensors)} tensors, {n_reinterpreted} dtype-fixed, "
              f"{n_transposed} transposed, {n_split} split")

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # Copy and clean config (remove PaddleFleet-specific fields)
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f_cfg:
            hf_config = json.load(f_cfg)
        # Remove non-standard fields that may confuse HF
        for field in ["attn_type"]:
            hf_config.pop(field, None)
        with open(output_path / "config.json", "w") as f_out:
            json.dump(hf_config, f_out, indent=2)
        print(f"  Saved config (cleaned)")

    # Copy tokenizer files
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tokenizer_files = [
            "tokenizer_config.json", "tokenizer.json", "vocab.json",
            "merges.txt", "special_tokens_map.json", "added_tokens.json",
        ]
        copied = []
        for fname in tokenizer_files:
            src = Path(tokenizer_dir) / fname
            if src.exists():
                shutil.copy2(src, output_path / fname)
                copied.append(fname)
        print(f"  Copied tokenizer: {', '.join(copied)}")

    # Validate
    print("\n  Validation:")
    check_file = str(output_path / shard_files[0].name)
    f_check = safe_open(check_file, framework="pt")
    sample_keys = ["model.layers.0.self_attn.q_proj.weight",
                   "model.embed_tokens.weight",
                   "model.layers.0.mlp.gate_proj.weight"]
    for sk in sample_keys:
        if sk in f_check.keys():
            t = f_check.get_tensor(sk)
            stats = t.float()
            print(f"    {sk}: dtype={t.dtype}, shape={list(t.shape)}, "
                  f"mean={stats.mean().item():.6f}, std={stats.std().item():.6f}")

    print(f"\n  Done. HF checkpoint saved to: {output_dir}")
    print(f"  Total keys: {len(weight_map)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge PaddleFleet distcp shards and convert to HuggingFace format"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to model_state directory (with .distcp files) or already-merged directory (with safetensors)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output HF checkpoint directory"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to model config.json"
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to tokenizer directory (optional)"
    )
    parser.add_argument(
        "--merged-dir", default=None,
        help="Intermediate directory for merged safetensors (default: <input>_merged)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Determine if we need to merge
    if needs_merge(args.input):
        if args.merged_dir:
            merged_dir = args.merged_dir
        else:
            parent = Path(args.input).parent
            basename = Path(args.input).name
            merged_dir = str(parent / f"{basename}_merged")

        merge_distcp(args.input, merged_dir)
        safetensors_dir = merged_dir
    else:
        print("Input already contains safetensors files, skipping merge step.")
        safetensors_dir = args.input

    # Convert to HF format
    convert_to_hf(safetensors_dir, args.output, args.config, args.tokenizer)


if __name__ == "__main__":
    main()
