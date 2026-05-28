"""Read per-layer KV grouping from VHA conversion diagnostics.

The conversion script `convert_gqa_to_vha_activation.py` already performed
per-layer exhaustive attention-aware balanced grouping search and stored the
result in conversion_diagnostics.json. We reuse those groupings rather than
re-running the search.

Output format:
    groupings: List[List[List[int]]]
        groupings[layer_idx] = [group_0_heads, group_1_heads]
        e.g. groupings[0] = [[0, 3, 4, 7], [1, 2, 5, 6]]
"""
import json
from pathlib import Path
from typing import List


def load_groupings(diagnostics_path: str | Path) -> List[List[List[int]]]:
    """Load per-layer grouping from a conversion_diagnostics.json file."""
    diagnostics_path = Path(diagnostics_path)
    with open(diagnostics_path) as f:
        data = json.load(f)

    layers_block = data.get("layers")
    if layers_block is None:
        raise KeyError(
            f"`layers` field not found in {diagnostics_path}; "
            "is this really a conversion_diagnostics.json?"
        )

    # layers may be either a list (indexed by layer) or a dict {layer_idx: ...}
    if isinstance(layers_block, dict):
        items = sorted(layers_block.items(), key=lambda kv: int(kv[0]))
        layers = [v for _, v in items]
    else:
        layers = layers_block

    groupings: List[List[List[int]]] = []
    for layer_idx, layer in enumerate(layers):
        group_errors = layer.get("group_errors") or []
        head_groups: List[List[int]] = []
        for entry in group_errors:
            if "k_src_heads" not in entry:
                # search_summary entry — skip
                continue
            head_groups.append(list(entry["k_src_heads"]))
        if len(head_groups) == 0:
            raise ValueError(f"Layer {layer_idx}: no k_src_heads found")
        groupings.append(head_groups)

    return groupings


def validate_groupings(
    groupings: List[List[List[int]]], n_layers: int, n_src_heads: int, n_groups: int
) -> None:
    """Sanity check: every layer has n_groups balanced groups partitioning n_src_heads."""
    assert (
        len(groupings) == n_layers
    ), f"expected {n_layers} layers, got {len(groupings)}"
    expected_per_group = n_src_heads // n_groups
    for li, layer_groups in enumerate(groupings):
        assert (
            len(layer_groups) == n_groups
        ), f"layer {li}: expected {n_groups} groups, got {len(layer_groups)}"
        flat = [h for g in layer_groups for h in g]
        assert sorted(flat) == list(range(n_src_heads)), (
            f"layer {li}: groups do not partition heads 0..{n_src_heads - 1}: {layer_groups}"
        )
        for gi, g in enumerate(layer_groups):
            assert (
                len(g) == expected_per_group
            ), f"layer {li} group {gi}: expected size {expected_per_group}, got {len(g)}"


def grouping_to_assignment(layer_groups: List[List[int]], n_src_heads: int) -> List[int]:
    """Convert [[0,3,4,7],[1,2,5,6]] → [0,1,1,0,0,1,1,0]   (head h → group idx).

    Used to build initial fusion weights ω_K[h, g] = 1/|group_g| if h ∈ group_g.
    """
    assignment = [-1] * n_src_heads
    for gi, heads in enumerate(layer_groups):
        for h in heads:
            assignment[h] = gi
    if -1 in assignment:
        raise ValueError(f"Some head not assigned: {layer_groups}")
    return assignment


if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/"
        "PaddleFleet/Research/VHA-Warmup/output/"
        "qwen3_vha_1p7B_kv_postmix_activation_128/conversion_diagnostics.json"
    )
    g = load_groupings(path)
    validate_groupings(g, n_layers=28, n_src_heads=8, n_groups=2)
    print(f"Loaded {len(g)} layers; example layer 0 = {g[0]}")
    for li in range(0, 28, 7):
        print(f"  layer {li:2d}: {g[li]}  → assignment {grouping_to_assignment(g[li], 8)}")
