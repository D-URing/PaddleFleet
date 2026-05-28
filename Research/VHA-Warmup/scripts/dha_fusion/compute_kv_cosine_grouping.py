"""DHA-paper-aligned grouping & omega init via K/V projection cosine similarity."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

N_LAYERS = 28
HIDDEN = 2048
N_Q_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
Q_PER_KV = N_Q_HEADS // N_KV_HEADS
N_GROUPS = 2
GROUP_SIZE = N_KV_HEADS // N_GROUPS
GROUP_DIM = (Q_PER_KV + 2) * HEAD_DIM
Q_DIM = Q_PER_KV * HEAD_DIM


def split_qkv_per_head(qkv_w: np.ndarray):
    grouped = qkv_w.reshape(HIDDEN, N_KV_HEADS, GROUP_DIM)
    K = grouped[:, :, Q_DIM : Q_DIM + HEAD_DIM]
    V = grouped[:, :, Q_DIM + HEAD_DIM :]
    K = np.transpose(K, (1, 0, 2)).astype(np.float32)
    V = np.transpose(V, (1, 0, 2)).astype(np.float32)
    return K, V


def head_cosine_matrix(per_head: np.ndarray) -> np.ndarray:
    flat = per_head.reshape(N_KV_HEADS, -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    flat_n = flat / norm
    return flat_n @ flat_n.T


def best_balanced_partition(cos_combined: np.ndarray):
    heads = list(range(N_KV_HEADS))
    best = None
    for tail in itertools.combinations(heads[1:], GROUP_SIZE - 1):
        g0 = tuple(sorted((0,) + tail))
        g1 = tuple(h for h in heads if h not in g0)

        def avg_intra(g):
            if len(g) <= 1:
                return 0.0
            s = 0.0
            cnt = 0
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    s += cos_combined[g[i], g[j]]
                    cnt += 1
            return s / cnt

        score = avg_intra(g0) + avg_intra(g1)
        if best is None or score > best[0]:
            best = (score, [list(g0), list(g1)])
    return best[1], best[0]


def cosine_weighted_init_logits(per_head: np.ndarray, groups, temperature: float = 5.0):
    flat = per_head.reshape(N_KV_HEADS, -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    flat_n = flat / norm
    logits = np.zeros((N_KV_HEADS, N_GROUPS), dtype=np.float32)
    for gi, members in enumerate(groups):
        members = list(members)
        centroid = flat_n[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        for h in members:
            logits[h, gi] = float(temperature * (flat_n[h] @ centroid))
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=5.0)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    files = sorted(ckpt_dir.glob("model-*.safetensors"))
    assert len(files) == 1, f"expected 1 safetensors, got {files}"

    out = {
        "n_layers": N_LAYERS, "n_kv_heads": N_KV_HEADS, "n_groups": N_GROUPS,
        "head_dim": HEAD_DIM, "temperature": args.temperature, "layers": [],
    }

    with safe_open(files[0], framework="pt") as f:
        for li in range(N_LAYERS):
            key = f"layers.{li}.self_attn.qkv_proj.weight"
            t = f.get_tensor(key).to(dtype=torch.float32).cpu().numpy()
            assert t.shape == (HIDDEN, N_KV_HEADS * GROUP_DIM)

            K_per, V_per = split_qkv_per_head(t)
            cos_k = head_cosine_matrix(K_per)
            cos_v = head_cosine_matrix(V_per)
            cos_combined = 0.5 * (cos_k + cos_v)

            groups, score = best_balanced_partition(cos_combined)
            wk = cosine_weighted_init_logits(K_per, groups, args.temperature)
            wv = cosine_weighted_init_logits(V_per, groups, args.temperature)

            def avg_intra(mat, members):
                if len(members) <= 1:
                    return 0.0
                vals = [mat[a, b] for a in members for b in members if a < b]
                return float(np.mean(vals))

            intra_k = [avg_intra(cos_k, g) for g in groups]
            intra_v = [avg_intra(cos_v, g) for g in groups]

            out["layers"].append({
                "layer": li, "groups": groups, "score": float(score),
                "cos_k": cos_k.tolist(), "cos_v": cos_v.tolist(),
                "omega_k_init_logits": wk.tolist(),
                "omega_v_init_logits": wv.tolist(),
                "intra_group_cos_k": intra_k,
                "intra_group_cos_v": intra_v,
            })

            print(f"L{li:2d}: groups={groups}  score={score:+.4f}  "
                  f"intra_k={intra_k[0]:+.3f}/{intra_k[1]:+.3f}  "
                  f"intra_v={intra_v[0]:+.3f}/{intra_v[1]:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")

    diag_path = ("/root/paddlejob/share-storage/gpfs/system-public/dingxibo/"
                 "PaddleFleet/Research/VHA-Warmup/output/"
                 "qwen3_vha_1p7B_kv_postmix_activation_128/conversion_diagnostics.json")
    if Path(diag_path).exists():
        old = json.load(open(diag_path))
        old_layers = old["layers"]
        if isinstance(old_layers, dict):
            old_layers = [old_layers[str(i)] for i in range(N_LAYERS)]
        same = 0
        for li in range(N_LAYERS):
            old_g = sorted([sorted(e["k_src_heads"]) for e in old_layers[li]["group_errors"]
                           if "k_src_heads" in e])
            new_g = sorted([sorted(g) for g in out["layers"][li]["groups"]])
            if old_g == new_g:
                same += 1
        print(f"Layers with same partition as postmix grouping: {same}/{N_LAYERS}")


if __name__ == "__main__":
    main()
