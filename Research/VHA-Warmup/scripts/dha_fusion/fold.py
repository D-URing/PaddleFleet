#!/usr/bin/env python3
"""Fold a DHA-fusion ckpt into a standard VHA(2-KV-head + postmix) ckpt.

Reads:
  - fusion-mode safetensors (still has 8-head K_proj/V_proj + dha_fusion.* params)
  - dha_fusion_meta.json (groupings, dims)

Writes:
  - VHA ckpt: K_proj/V_proj collapsed 8->2 via hard 1/|group| averaging
  - postmix_U / postmix_V kept as-is (rename to match VHA expected names if needed)

Math:
  K_proj.weight ckpt shape: [hidden, n_kv_src * head_dim] = [2048, 8*128]
    reshape -> [2048, 8, 128]
    For each target group g: avg over heads in g -> [2048, 1, 128]
    stack groups -> [2048, 2, 128] -> [2048, 2*128]

  Optionally, omega_K from fusion is used to detect convergence (logit gap
  between in-group and out-of-group columns); if not converged we WARN.
"""
import argparse
import json
import os
import sys
import numpy as np


def ckpt_key_to_pipeline_name(key):
    k = key.replace("model.", "", 1) if key.startswith("model.") else key
    if k.startswith("embedding."):
        return f"0.{k}"
    if k.startswith("layers."):
        parts = k.split(".", 2)
        layer_idx = int(parts[1])
        return f"{layer_idx + 1}.{parts[2]}"
    if k == "norm.weight":
        return "29.norm.weight"
    if k == "lm_head.weight":
        return "30.weight"
    return None


def pipeline_name_to_hf_key(pname):
    """Inverse of ckpt_key_to_pipeline_name. Returns model.* HF key."""
    parts = pname.split(".", 1)
    try:
        idx = int(parts[0])
    except ValueError:
        return pname
    rest = parts[1] if len(parts) > 1 else ""
    if idx == 0:
        return f"model.{rest}"
    if idx == 29:
        return "model.norm.weight"
    if idx == 30:
        return "model.lm_head.weight"
    return f"model.layers.{idx - 1}.{rest}"


def collapse_kv_weight(w, head_dim, groupings_layer):
    """w: numpy [hidden, n_src_heads * head_dim]. Returns [hidden, n_groups * head_dim]."""
    hidden = w.shape[0]
    n_src = w.shape[1] // head_dim
    n_groups = len(groupings_layer)
    w = w.reshape(hidden, n_src, head_dim)
    out = np.zeros((hidden, n_groups, head_dim), dtype=w.dtype)
    for g, members in enumerate(groupings_layer):
        out[:, g, :] = w[:, members, :].mean(axis=1)
    return out.reshape(hidden, n_groups * head_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--omega_threshold", type=float, default=0.85,
                        help="Min in-group omega softmax weight per head to consider converged.")
    args = parser.parse_args()

    from safetensors import safe_open
    from safetensors.numpy import save_file
    import ml_dtypes

    meta_path = os.path.join(args.fusion_ckpt, "dha_fusion_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    groupings = meta["groupings"]
    head_dim = meta["head_dim"]
    n_groups = meta["n_groups"]
    print(f"[fold] groupings: {len(groupings)} layers, n_groups={n_groups}, head_dim={head_dim}")

    # Read all tensors
    sf_files = sorted(f for f in os.listdir(args.fusion_ckpt) if f.endswith(".safetensors"))
    tensors = {}
    for sf in sf_files:
        with safe_open(os.path.join(args.fusion_ckpt, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).float().numpy()
    print(f"[fold] loaded {len(tensors)} tensors")

    # Convergence check: omega columns should be nearly one-hot within each group
    print(f"[fold] checking omega convergence (threshold {args.omega_threshold}):")
    not_converged = []
    for layer_idx in range(len(groupings)):
        for kv in ("K", "V"):
            # Pipeline-named: f"{layer_idx+1}.self_attn.dha_fusion.omega_{kv}_logits"
            # HF-named (after our save): f"model.layers.{layer_idx}.self_attn.dha_fusion.omega_{kv}_logits"
            candidates = [
                f"model.layers.{layer_idx}.self_attn.dha_fusion.omega_{kv}_logits",
                f"{layer_idx + 1}.self_attn.dha_fusion.omega_{kv}_logits",
            ]
            omega_logits = None
            for c in candidates:
                if c in tensors:
                    omega_logits = tensors[c]
                    break
            if omega_logits is None:
                continue
            # omega_logits shape [n_src=8, n_groups=2]; apply per-column masked softmax
            # using groupings to detect non-convergence
            import math
            for g, members in enumerate(groupings[layer_idx]):
                col = omega_logits[:, g].copy()
                # mask non-members to -inf
                masked = np.full_like(col, -1e9)
                masked[members] = col[members]
                e = np.exp(masked - masked.max())
                w = e / e.sum()
                # check max in-group weight
                in_g = w[members]
                max_in_g = in_g.max()
                if max_in_g < args.omega_threshold:
                    not_converged.append((layer_idx, kv, g, max_in_g))
    if not_converged:
        print(f"[fold] WARN: {len(not_converged)} omega groups not converged (showing first 10):")
        for L, kv, g, w in not_converged[:10]:
            print(f"  layer {L} {kv} group {g}: max omega={w:.3f}")
    else:
        print(f"[fold] omega converged for all groups")

    # Build output tensors: collapse K_proj/V_proj for each layer; pass through everything else
    out_tensors = {}
    for key, val in tensors.items():
        # Identify K_proj / V_proj weights
        # HF naming: model.layers.{L}.self_attn.linear_kv.weight (paddlefleet GQA)
        # Or: model.layers.{L}.self_attn.linear_k.weight + linear_v.weight (separated)
        # Or: model.layers.{L}.self_attn.k_proj.weight (other naming)
        is_k = False
        is_v = False
        layer_idx = None
        for prefix_pat in ("model.layers.",):
            if key.startswith(prefix_pat):
                rest = key[len(prefix_pat):]
                try:
                    layer_idx = int(rest.split(".", 1)[0])
                except ValueError:
                    layer_idx = None
                break
        if layer_idx is not None:
            tail = key.rsplit(".", 1)[0]
            if tail.endswith(".linear_k") or tail.endswith(".k_proj"):
                is_k = True
            elif tail.endswith(".linear_v") or tail.endswith(".v_proj"):
                is_v = True

        if (is_k or is_v) and key.endswith(".weight"):
            n_src = val.shape[1] // head_dim
            if n_src == 8 and layer_idx < len(groupings):
                collapsed = collapse_kv_weight(val, head_dim, groupings[layer_idx])
                out_tensors[key] = collapsed
                print(f"[fold] collapsed {key}: {val.shape} -> {collapsed.shape}")
                continue

        # Drop omega params (no longer needed after fold)
        if "dha_fusion.omega" in key:
            continue
        # Keep postmix params with same name (caller may need to rename for VHA model)
        out_tensors[key] = val

    # Cast to bf16 for save
    out_bf16 = {k: v.astype(ml_dtypes.bfloat16) for k, v in out_tensors.items()}

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, "model-00001-of-00001.safetensors")
    save_file(out_bf16, out_file)
    weight_map = {k: "model-00001-of-00001.safetensors" for k in out_bf16.keys()}
    total_size = sum(t.nbytes for t in out_bf16.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=4)

    # Patch config.json (n_kv_heads -> n_groups)
    src_config_path = os.path.join(args.fusion_ckpt, "config.json")
    if os.path.exists(src_config_path):
        with open(src_config_path) as f:
            config = json.load(f)
        # Common keys: num_key_value_heads (HF), n_kv_heads (some forks)
        for k in ("num_key_value_heads", "n_kv_heads"):
            if k in config:
                config[k] = n_groups
        # Mark postmix presence
        config["dha_postmix_rank"] = meta.get("postmix_rank", 4)
        with open(os.path.join(args.output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print(f"[fold] saved VHA ckpt to {args.output_path}")
    print(f"[fold] {len(out_bf16)} tensors, {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
