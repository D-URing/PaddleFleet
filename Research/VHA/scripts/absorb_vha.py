#!/usr/bin/env python3
"""
Absorb VHA parameters (premix, postmix) into standard GQA weights.

After absorption, the model becomes a standard GQA model with:
  - num_attention_heads = H_k * H_q (expanded virtual heads)
  - num_key_value_heads = H_k (unchanged)
  - No premix/postmix parameters

Absorption math:
  Premix into W_q:
    Original: Q_expanded[k,h,:] = (hidden @ W_q[h]) @ W_premix[k]
    Absorbed: W_q_new[k,h] = W_q[h] @ W_premix[k]   (shape [D, d])
    New Q has H_k * H_q heads total.

  Postmix into W_o:
    Original: x_out[h] = sum_j (I + V @ U^T)[h,j] * x[j]
    Mixing matrix M = I + V @ U^T, shape [total_heads, total_heads]
    Absorbed: W_o_new[h] = sum_j M[h,j] * W_o[j]

Usage:
    python scripts/absorb_vha.py \
        --input ./output/qwen3_vha_4B_pretrain/checkpoint-24000/model_state \
        --output ./ckpts/qwen3_vha_4B_hf \
        --config ./config/qwen3/Qwen3-4B-VHA/config.json \
        --tokenizer ./config/qwen3/tokenizer
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def fix_dtype(tensor):
    """Reinterpret uint16 as bfloat16."""
    if tensor.dtype == torch.uint16:
        return tensor.view(torch.bfloat16)
    return tensor


def absorb_and_convert(merged_dir: str, output_dir: str, config_path: str, tokenizer_dir: str):
    """
    Absorb VHA premix/postmix into Q/O projections and convert to HF GQA format.
    """
    print("=" * 60)
    print("  VHA Absorption + Conversion to HF GQA")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Load VHA config
    with open(config_path) as f:
        config = json.load(f)

    H_q = config["num_attention_heads"]       # pre-expansion Q heads (e.g. 16)
    H_k = config["num_key_value_heads"]       # KV heads (e.g. 2)
    d = config.get("head_dim", config["hidden_size"] // H_q)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    tie_word_embeddings = config.get("tie_word_embeddings", False)
    total_heads = H_k * H_q                   # expanded virtual heads (e.g. 32)

    print(f"  VHA Config: H_q={H_q}, H_k={H_k}, d={d}, hidden={hidden_size}")
    print(f"  After absorption: num_attention_heads={total_heads}, num_kv_heads={H_k}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_path = Path(merged_dir)
    shard_files = sorted(merged_path.glob("model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No model-*.safetensors in {merged_dir}")

    # First pass: collect all tensors (need premix/postmix from same layer as qkv/o)
    print(f"  Loading {len(shard_files)} shard(s)...")
    all_tensors = {}
    for sf in shard_files:
        f = safe_open(str(sf), framework="pt")
        for key in f.keys():
            all_tensors[key] = fix_dtype(f.get_tensor(key))

    print(f"  Total keys loaded: {len(all_tensors)}")

    # Identify layers
    num_layers = config["num_hidden_layers"]
    new_tensors = {}
    n_absorbed_premix = 0
    n_absorbed_postmix = 0

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}.self_attn."

        # --- Get VHA-specific weights ---
        premix_key = prefix + "vha_premix_weight"
        postmix_U_key = prefix + "vha_postmix_U"
        postmix_V_key = prefix + "vha_postmix_V"

        has_premix = premix_key in all_tensors
        has_postmix = postmix_U_key in all_tensors and postmix_V_key in all_tensors

        # --- Get QKV fused weight ---
        qkv_key = prefix + "qkv_proj.weight"
        o_key = prefix + "o_proj.weight"

        if qkv_key not in all_tensors:
            raise KeyError(f"Missing {qkv_key}")
        if o_key not in all_tensors:
            raise KeyError(f"Missing {o_key}")

        # QKV weight: Paddle layout [hidden_size, (H_q + 2*H_k) * d] for GQA grouped
        # But VHA has different Q size. Let's check the actual shape.
        qkv_w = all_tensors[qkv_key]  # [hidden_size, qkv_out_dim]

        # In VHA, the QKV proj outputs:
        #   Q: H_q heads * d dims
        #   K: H_k heads * d dims
        #   V: H_k heads * d dims
        # Grouped layout: [G0_Q(H_q/H_k heads), G0_K(1 head), G0_V(1 head), G1_Q, G1_K, G1_V, ...]
        # Each group: (H_q/H_k + 2) * d dims
        heads_per_group = H_q // H_k
        group_dim = (heads_per_group + 2) * d
        expected_qkv_dim = H_k * group_dim
        assert qkv_w.shape[1] == expected_qkv_dim, \
            f"Layer {layer_idx}: qkv shape {qkv_w.shape}, expected [:, {expected_qkv_dim}]"

        # Split QKV from grouped layout (Paddle layout: [hidden_size, out_dim])
        # Reshape to [hidden_size, H_k, group_dim]
        qkv_grouped = qkv_w.reshape(hidden_size, H_k, group_dim)
        q_dim_per_group = heads_per_group * d

        # Q: [hidden_size, H_k, heads_per_group * d]
        q_per_group = qkv_grouped[:, :, :q_dim_per_group]
        # K: [hidden_size, H_k, d]
        k_per_group = qkv_grouped[:, :, q_dim_per_group:q_dim_per_group + d]
        # V: [hidden_size, H_k, d]
        v_per_group = qkv_grouped[:, :, q_dim_per_group + d:]

        # q_per_group: [hidden_size, H_k, heads_per_group, d]
        q_per_group = q_per_group.reshape(hidden_size, H_k, heads_per_group, d)

        # --- Absorb premix into Q ---
        if has_premix:
            # premix_weight: [H_k, d, d]
            W_premix = all_tensors[premix_key].float()  # [H_k, d, d]

            # For each KV group k, for each Q head h:
            #   W_q_new[k, h] = W_q[h_within_group] @ W_premix[k]
            # But premix expands EACH of the H_q heads by EACH of the H_k groups:
            #   Q_expanded[k, h, :] = Q[h, :] @ W_premix[k, :, :]
            # where Q[h] comes from group (h // heads_per_group), head (h % heads_per_group)
            #
            # Wait - re-read the code:
            #   query shape: [b, t, H_q, d] (all H_q heads, not per-group)
            #   premix: einsum("bthd,kde->btkhe", query, W_premix)
            #   result: [b, t, H_k, H_q, d]
            #
            # So premix takes ALL H_q heads and expands each by H_k groups.
            # W_q_new[k, h] = W_q_original[h] @ W_premix[k]
            #
            # Original Q heads from grouped layout:
            #   Q head h belongs to group (h // heads_per_group)
            #   Within that group, it's head (h % heads_per_group)
            #
            # After premix, we have H_k * H_q total heads.
            # New head index (k, h) where k in [0, H_k), h in [0, H_q):
            #   W_q_new[k, h] = W_q_original[h] @ W_premix[k]
            #
            # W_q_original[h] is in Paddle layout: [hidden_size, d]
            # It comes from group (h // heads_per_group), position (h % heads_per_group)

            # Flatten all Q heads: [hidden_size, H_q, d]
            # From grouped: q_per_group is [hidden_size, H_k, heads_per_group, d]
            # Original Q head ordering: group 0 heads 0..hpg-1, group 1 heads 0..hpg-1, ...
            # So Q head h = group[h // hpg][h % hpg]
            # Flatten: [hidden_size, H_k * heads_per_group, d] = [hidden_size, H_q, d]
            W_q_all = q_per_group.reshape(hidden_size, H_q, d).float()

            # Absorb: W_q_new[k, h, :, :] = W_q_all[:, h, :] @ W_premix[k, :, :]
            # W_q_all: [hidden_size, H_q, d]
            # W_premix: [H_k, d, d]
            # Result: [hidden_size, H_k, H_q, d]
            W_q_new = torch.einsum("ihd,kde->ikhe", W_q_all, W_premix)
            # W_q_new: [hidden_size, H_k, H_q, d]
            # Flatten to [hidden_size, total_heads, d]
            W_q_new = W_q_new.reshape(hidden_size, total_heads, d)

            n_absorbed_premix += 1
        else:
            # No premix - just use original Q (already H_q heads per group)
            # Expand to total_heads by repeating each Q head for each KV group
            # This shouldn't happen for VHA, but handle gracefully
            W_q_new = q_per_group.reshape(hidden_size, H_q, d).float()
            W_q_new = W_q_new.unsqueeze(1).expand(-1, H_k, -1, -1)
            W_q_new = W_q_new.reshape(hidden_size, total_heads, d)

        # --- Handle O_proj with postmix absorption ---
        o_w = all_tensors[o_key]  # Paddle layout: [total_heads * d, hidden_size]

        if has_postmix:
            U = all_tensors[postmix_U_key].float()  # [total_heads, r]
            V = all_tensors[postmix_V_key].float()  # [total_heads, r]

            # Mixing matrix M = I + V @ U^T, shape [total_heads, total_heads]
            M = torch.eye(total_heads, dtype=torch.float32) + V @ U.T

            # O_proj weight: Paddle layout [total_heads * d, hidden_size]
            # Reshape to [total_heads, d, hidden_size]
            W_o = o_w.float().reshape(total_heads, d, hidden_size)

            # Absorb: W_o_new[j] = sum_h M[h,j] * W_o[h]  (i.e. M^T @ W_o)
            # M: [total_heads, total_heads], W_o: [total_heads, d, hidden_size]
            W_o_new = torch.einsum("hj,hde->jde", M, W_o)
            # Flatten back: [total_heads * d, hidden_size]
            W_o_new = W_o_new.reshape(total_heads * d, hidden_size)

            n_absorbed_postmix += 1
        else:
            W_o_new = o_w.float()

        # --- Build new QKV in GQA grouped layout for the absorbed model ---
        # New model: total_heads Q heads, H_k KV heads
        # New heads_per_group = total_heads / H_k = H_q
        # Grouped layout: [G0_Q(H_q heads), G0_K(1), G0_V(1), G1_Q(H_q), G1_K(1), G1_V(1), ...]
        #
        # W_q_new: [hidden_size, H_k, H_q, d] (already in k,h order)
        # K: [hidden_size, H_k, d]
        # V: [hidden_size, H_k, d]

        K_w = k_per_group.float()  # [hidden_size, H_k, d]
        V_w = v_per_group.float()  # [hidden_size, H_k, d]
        W_q_grouped = W_q_new.reshape(hidden_size, H_k, H_q, d)

        # Build grouped: for each group k: [Q(H_q*d), K(d), V(d)]
        new_group_dim = (H_q + 2) * d
        qkv_new = torch.zeros(hidden_size, H_k, new_group_dim, dtype=torch.float32)
        qkv_new[:, :, :H_q * d] = W_q_grouped.reshape(hidden_size, H_k, H_q * d)
        qkv_new[:, :, H_q * d:H_q * d + d] = K_w
        qkv_new[:, :, H_q * d + d:] = V_w

        # Flatten: [hidden_size, H_k * new_group_dim]
        qkv_new = qkv_new.reshape(hidden_size, H_k * new_group_dim)

        # Store absorbed weights (still in Paddle layout)
        all_tensors[qkv_key] = qkv_new.to(torch.bfloat16)
        all_tensors[o_key] = W_o_new.to(torch.bfloat16)

        # Remove VHA-specific keys
        for vha_key in [premix_key, postmix_U_key, postmix_V_key]:
            all_tensors.pop(vha_key, None)

    print(f"  Absorbed premix in {n_absorbed_premix} layers")
    print(f"  Absorbed postmix in {n_absorbed_postmix} layers")

    # --- Now convert to HF format (same as merge_and_convert.py Step 2) ---
    print("\n  Converting to HF format...")

    # After absorption, the model is GQA with:
    #   num_attention_heads = total_heads = H_k * H_q
    #   num_key_value_heads = H_k
    #   heads_per_group = H_q (the original num_attention_heads)
    new_heads_per_group = H_q
    new_num_heads = total_heads
    new_num_kv_heads = H_k

    TRANSPOSE_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                      "up_proj", "down_proj", "qkv_proj", "up_gate_proj"]

    def should_transpose(key):
        for tk in TRANSPOSE_KEYS:
            if f".{tk}.weight" in key or key == f"{tk}.weight":
                return True
        return False

    new_hf_tensors = {}
    n_transposed = 0
    n_split = 0

    for key, tensor in all_tensors.items():
        # Transpose linear weights
        if tensor.ndim == 2 and should_transpose(key):
            tensor = tensor.T.contiguous()
            n_transposed += 1

        # Map key names
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
            # After absorption: grouped layout with new_heads_per_group = H_q per group
            # Transposed: [H_k * (H_q + 2) * d, hidden_size]
            group_dim_new = (new_heads_per_group + 2) * d
            grouped = tensor.reshape(new_num_kv_heads, group_dim_new, hidden_size)
            q_dim = new_heads_per_group * d
            q = grouped[:, :q_dim, :].reshape(new_num_heads * d, hidden_size)
            k = grouped[:, q_dim:q_dim + d, :].reshape(new_num_kv_heads * d, hidden_size)
            v = grouped[:, q_dim + d:, :].reshape(new_num_kv_heads * d, hidden_size)

            base = hf_key.replace("qkv_proj.weight", "")
            new_hf_tensors[base + "q_proj.weight"] = q
            new_hf_tensors[base + "k_proj.weight"] = k
            new_hf_tensors[base + "v_proj.weight"] = v
            n_split += 1
        elif ".up_gate_proj.weight" in key:
            gate, up = tensor.split([intermediate_size, intermediate_size], dim=0)
            base = hf_key.replace("up_gate_proj.weight", "")
            new_hf_tensors[base + "gate_proj.weight"] = gate
            new_hf_tensors[base + "up_proj.weight"] = up
            n_split += 1
        else:
            new_hf_tensors[hf_key] = tensor

    # Tied embeddings
    if tie_word_embeddings and "lm_head.weight" not in new_hf_tensors:
        if "model.embed_tokens.weight" in new_hf_tensors:
            new_hf_tensors["lm_head.weight"] = new_hf_tensors["model.embed_tokens.weight"]

    print(f"  {len(new_hf_tensors)} tensors, {n_transposed} transposed, {n_split} split")

    # Save
    shard_name = "model-00001-of-00001.safetensors"
    save_file(new_hf_tensors, str(output_path / shard_name))

    # Weight map
    weight_map = {}
    total_size = 0
    for k, t in new_hf_tensors.items():
        weight_map[k] = shard_name
        total_size += t.numel() * t.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(output_path / "model.safetensors.index.json", "w") as fp:
        json.dump(index, fp, indent=2)

    # Save absorbed GQA config
    hf_config = dict(config)
    hf_config["num_attention_heads"] = total_heads
    hf_config["num_key_value_heads"] = H_k
    # Remove VHA-specific fields
    for field in ["attn_type", "vha_enable_premix", "vha_enable_postmix", "vha_postmix_rank",
                  "vha_premix_init_alpha"]:
        hf_config.pop(field, None)
    with open(output_path / "config.json", "w") as f_out:
        json.dump(hf_config, f_out, indent=2)
    print(f"  Config: num_attention_heads={total_heads}, num_kv_heads={H_k}")

    # Copy tokenizer
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json",
                           "merges.txt", "special_tokens_map.json", "added_tokens.json"]
        copied = []
        for fname in tokenizer_files:
            src = Path(tokenizer_dir) / fname
            if src.exists():
                shutil.copy2(src, output_path / fname)
                copied.append(fname)
        print(f"  Copied tokenizer: {', '.join(copied)}")

    # Validation
    print("\n  Validation:")
    sample_keys = ["model.layers.0.self_attn.q_proj.weight",
                   "model.layers.0.self_attn.o_proj.weight",
                   "model.embed_tokens.weight"]
    for sk in sample_keys:
        if sk in new_hf_tensors:
            t = new_hf_tensors[sk]
            stats = t.float()
            print(f"    {sk}: shape={list(t.shape)}, mean={stats.mean():.6f}, std={stats.std():.6f}")

    print(f"\n  Done. Absorbed HF checkpoint: {output_dir}")
    print(f"  Total keys: {len(new_hf_tensors)}")


def main():
    parser = argparse.ArgumentParser(
        description="Absorb VHA premix/postmix into GQA weights and convert to HF format"
    )
    parser.add_argument("--input", required=True,
                        help="Path to model_state dir (distcp) or merged dir (safetensors)")
    parser.add_argument("--output", required=True,
                        help="Output HF checkpoint directory")
    parser.add_argument("--config", required=True,
                        help="VHA model config.json")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer directory")
    parser.add_argument("--merged-dir", default=None,
                        help="Intermediate merged dir (default: <input>_merged)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    # Check if merge is needed
    input_path = Path(args.input)
    has_safetensors = list(input_path.glob("model-*.safetensors"))
    has_distcp = list(input_path.glob("*.distcp"))

    if has_safetensors:
        merged_dir = args.input
    elif has_distcp:
        # Need to merge first
        if args.merged_dir:
            merged_dir = args.merged_dir
        else:
            parent = input_path.parent
            merged_dir = str(parent / f"{input_path.name}_merged")

        print("=" * 60)
        print("  Step 0: Merging distcp shards")
        print(f"  Input:  {args.input}")
        print(f"  Output: {merged_dir}")
        print("=" * 60)

        import paddle
        from paddle.distributed.flex_checkpoint.dcp.load_state_dict import merge_sharded_state_dict

        os.makedirs(merged_dir, exist_ok=True)
        merge_sharded_state_dict(
            load_path=args.input,
            save_path=merged_dir,
            prefix="model",
            safetensor_prefix="model",
            offload=True,
        )
        print(f"  Merge complete.\n")
    else:
        raise FileNotFoundError(f"No .distcp or safetensors in {args.input}")

    absorb_and_convert(merged_dir, args.output, args.config, args.tokenizer)


if __name__ == "__main__":
    main()
