#!/usr/bin/env python3
"""
Convert PaddleFleet unified_checkpoint (safetensors) to HuggingFace-compatible format.

PaddleFleet saves bfloat16 weights as uint16 in safetensors, and uses Paddle's
Linear layout [in_features, out_features]. This script:
  1. Reinterprets uint16 tensors as bfloat16
  2. Transposes Linear weight matrices to PyTorch layout [out_features, in_features]
  3. Saves a clean HF-compatible checkpoint loadable by transformers

Usage:
    python scripts/convert_paddle_to_hf.py --input <paddle_ckpt_dir> --output <hf_ckpt_dir>

Examples:
    python scripts/convert_paddle_to_hf.py \
        --input /path/to/ckpts/qwen3_GQA_1p7B_pretrain \
        --output /path/to/ckpts/qwen3_GQA_1p7B_pretrain_hf

    python scripts/convert_paddle_to_hf.py \
        --input /path/to/ckpts/qwen3_GQA_1p7B_pretrain/checkpoint-24000 \
        --output /path/to/ckpts/qwen3_GQA_1p7B_ckpt24000_hf
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Keys whose .weight tensors need transposing (Paddle [in, out] -> PyTorch [out, in])
TRANSPOSE_WEIGHT_KEYS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Files to copy from source to destination
COPY_FILES = [
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "tokenizer.json",
]


def should_transpose(key: str) -> bool:
    """Check if a weight key should be transposed."""
    for trans_key in TRANSPOSE_WEIGHT_KEYS:
        if f".{trans_key}.weight" in key or key == f"{trans_key}.weight":
            return True
    return False


def fix_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret uint16 as bfloat16 (PaddleFleet saves bf16 as uint16 in safetensors)."""
    if tensor.dtype == torch.uint16:
        return tensor.view(torch.bfloat16)
    return tensor


def convert_checkpoint(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    shard_files = sorted(input_path.glob("model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No model-*.safetensors files found in {input_dir}")

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(shard_files)} shard(s)")
    print()

    weight_map = {}
    total_size = 0

    for shard_file in shard_files:
        shard_name = shard_file.name
        print(f"Processing {shard_name}...")

        f = safe_open(str(shard_file), framework="pt")
        new_tensors = {}
        n_reinterpreted = 0
        n_transposed = 0

        for key in f.keys():
            tensor = f.get_tensor(key)

            # Step 1: Fix dtype (uint16 -> bfloat16)
            original_dtype = tensor.dtype
            tensor = fix_dtype(tensor)
            if original_dtype == torch.uint16:
                n_reinterpreted += 1

            # Step 2: Transpose linear weights
            if tensor.ndim == 2 and should_transpose(key):
                tensor = tensor.T.contiguous()
                n_transposed += 1

            new_tensors[key] = tensor
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()

        # Save converted shard
        save_file(new_tensors, str(output_path / shard_name))
        print(f"  {len(new_tensors)} tensors, {n_reinterpreted} dtype-fixed, {n_transposed} transposed")

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # Copy config and tokenizer files
    copied = []
    for fname in COPY_FILES:
        src = input_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            copied.append(fname)
    print(f"\nCopied: {', '.join(copied)}")

    # Validate: quick dtype check on first tensor
    print("\nValidation:")
    check_file = str(output_path / shard_files[0].name)
    f_check = safe_open(check_file, framework="pt")
    first_key = list(f_check.keys())[0]
    t = f_check.get_tensor(first_key)
    print(f"  {first_key}: dtype={t.dtype}, shape={list(t.shape)}")
    if t.dtype == torch.bfloat16:
        stats = t.float()
        print(f"  mean={stats.mean().item():.6f}, std={stats.std().item():.6f}")
        if abs(stats.mean().item()) < 1.0 and stats.std().item() < 1.0:
            print("  [OK] Values look like trained weights")
        else:
            print("  [WARNING] Values look abnormal, check conversion")
    elif t.dtype == torch.uint16:
        print("  [ERROR] Still uint16 after conversion!")
    else:
        print(f"  [OK] dtype={t.dtype}")

    print(f"\nDone. HF checkpoint saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PaddleFleet checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--input", required=True, help="Path to PaddleFleet checkpoint directory"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output HF checkpoint directory"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
