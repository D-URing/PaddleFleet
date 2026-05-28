"""DHA-style fusion attention patch (hook-based).

Treats DHA fusion as a transitional class between GQA and VHA:

    GQA(8 KV heads) -> [DHA fusion mode] -> VHA(2 KV heads + postmix)

During fusion training, GQA modeling is left untouched. We attach two hooks
onto each attention layer's `core_attention` submodule:

  forward_pre_hook(core_attention, (Q, K, V, ...))
      -> intercepts K, V (shape [B, T, 8, d]),
         caches them for ALM constraint computation,
         applies omega fusion -> K_fused, V_fused (shape [B, T, 2, d]),
         returns modified args.

  forward_post_hook(core_attention, _, attn_out)
      -> applies postmix UV residual on [B, T, 16*d] output. Postmix is a
         rank-r residual (default r=4) compensating for the rank deficit
         introduced by the 8 -> 2 KV contraction.

After the fusion phase converges, `fold.py` collapses omega to hard
1/|group| weights, averages K_proj / V_proj heads within groups (8 -> 2),
and saves a standard VHA checkpoint.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class FusionState(nn.Layer):
    """Per-layer trainable fusion params (omega gates + postmix UV residual).

    omega_{k,v}_init: optional [n_src, n_groups] ndarrays for cosine-weighted
    init. None -> Constant(0.0) -> uniform-after-softmax (mean-pooling start).
    """

    def __init__(
        self,
        head_assignment: Sequence[int],
        n_groups: int = 2,
        total_q_heads: int = 16,
        head_dim: int = 128,
        postmix_rank: int = 4,
        param_dtype: str = "float32",
        omega_k_init: Optional[np.ndarray] = None,
        omega_v_init: Optional[np.ndarray] = None,
    ):
        super().__init__()
        n_src = len(head_assignment)
        assert max(head_assignment) == n_groups - 1, (
            f"head_assignment max {max(head_assignment)} != n_groups-1 {n_groups-1}"
        )
        self.n_src = n_src
        self.n_groups = n_groups
        self.total_q_heads = total_q_heads
        self.head_dim = head_dim
        self.postmix_rank = postmix_rank

        mask = paddle.zeros([n_src, n_groups], dtype=param_dtype)
        for h, g in enumerate(head_assignment):
            mask[h, g] = 1.0
        self.register_buffer("fusion_mask", mask, persistable=False)
        self.group_members: List[List[int]] = [
            sorted([h for h, gg in enumerate(head_assignment) if gg == g])
            for g in range(n_groups)
        ]
        gs = paddle.to_tensor(
            [float(len(self.group_members[g])) for g in range(n_groups)],
            dtype=param_dtype,
        )
        self.register_buffer("group_size", gs, persistable=False)

        def _logit_init(init_arr):
            if init_arr is None:
                return nn.initializer.Constant(0.0)
            arr = np.asarray(init_arr, dtype=np.float32)
            if list(arr.shape) != [n_src, n_groups]:
                raise ValueError(
                    f"omega init shape {list(arr.shape)} != [{n_src}, {n_groups}]"
                )
            return nn.initializer.Assign(arr)

        self.omega_k_logits = self.create_parameter(
            shape=[n_src, n_groups],
            dtype=param_dtype,
            default_initializer=_logit_init(omega_k_init),
            is_bias=False,
        )
        self.omega_v_logits = self.create_parameter(
            shape=[n_src, n_groups],
            dtype=param_dtype,
            default_initializer=_logit_init(omega_v_init),
            is_bias=False,
        )

        # Postmix UV: U normal std=0.01, V zeros -> delta=0 at init.
        # Rank-r residual compensates for 8->2 KV contraction loss.
        self.postmix_U = self.create_parameter(
            shape=[total_q_heads, postmix_rank],
            dtype=param_dtype,
            default_initializer=nn.initializer.Normal(mean=0.0, std=0.01),
            is_bias=False,
        )
        self.postmix_V = self.create_parameter(
            shape=[total_q_heads, postmix_rank],
            dtype=param_dtype,
            default_initializer=nn.initializer.Constant(0.0),
            is_bias=False,
        )

        self.cached_k_pre_fusion: paddle.Tensor | None = None
        self.cached_v_pre_fusion: paddle.Tensor | None = None

    def _omega(self, logits: paddle.Tensor) -> paddle.Tensor:
        NEG_INF = -1e9
        masked = logits + (1.0 - self.fusion_mask) * NEG_INF
        return F.softmax(masked, axis=0)

    def omega_k(self) -> paddle.Tensor:
        return self._omega(self.omega_k_logits)

    def omega_v(self) -> paddle.Tensor:
        return self._omega(self.omega_v_logits)

    def fuse_kv(self, k: paddle.Tensor, v: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        wk = self.omega_k().cast(k.dtype)
        wv = self.omega_v().cast(v.dtype)
        k_fused = paddle.einsum("bthd,hg->btgd", k, wk)
        v_fused = paddle.einsum("bthd,hg->btgd", v, wv)
        return k_fused, v_fused

    def apply_postmix(self, attn_out: paddle.Tensor) -> paddle.Tensor:
        b, t, _ = attn_out.shape
        d = self.head_dim
        x = attn_out.reshape([b, t, self.total_q_heads, d])
        U = self.postmix_U.cast(x.dtype)
        V = self.postmix_V.cast(x.dtype)
        z = paddle.einsum("bthd,hr->btrd", x, U)
        delta = paddle.einsum("btrd,hr->bthd", z, V)
        return (x + delta).reshape([b, t, self.total_q_heads * d])


def _make_pre_hook(fs: FusionState, kv_arg_indices: Tuple[int, int] = (1, 2)):
    ki, vi = kv_arg_indices

    def pre_hook(layer, args):
        if len(args) <= max(ki, vi):
            raise RuntimeError(
                f"core_attention forward got {len(args)} args; expected at least "
                f"{max(ki, vi) + 1} (K at {ki}, V at {vi})."
            )
        k = args[ki]
        v = args[vi]
        fs.cached_k_pre_fusion = k
        fs.cached_v_pre_fusion = v
        k_fused, v_fused = fs.fuse_kv(k, v)
        new_args = list(args)
        new_args[ki] = k_fused
        new_args[vi] = v_fused
        return tuple(new_args)

    return pre_hook


def _make_post_hook(fs: FusionState):
    def post_hook(layer, args, output):
        if isinstance(output, tuple):
            attn_out = output[0]
            mixed = fs.apply_postmix(attn_out)
            return (mixed,) + output[1:]
        else:
            return fs.apply_postmix(output)
    return post_hook


def install_fusion_hooks(
    attention_layer: nn.Layer,
    head_assignment: Sequence[int],
    *,
    core_attention_attr: str = "core_attention",
    kv_arg_indices: Tuple[int, int] = (1, 2),
    n_groups: int = 2,
    total_q_heads: int = 16,
    head_dim: int = 128,
    postmix_rank: int = 4,
    omega_k_init: Optional[np.ndarray] = None,
    omega_v_init: Optional[np.ndarray] = None,
) -> FusionState:
    fs = FusionState(
        head_assignment=head_assignment,
        n_groups=n_groups,
        total_q_heads=total_q_heads,
        head_dim=head_dim,
        postmix_rank=postmix_rank,
        omega_k_init=omega_k_init,
        omega_v_init=omega_v_init,
    )
    attention_layer.add_sublayer("dha_fusion", fs)

    if not hasattr(attention_layer, core_attention_attr):
        raise AttributeError(
            f"Attention layer {type(attention_layer).__name__} has no submodule "
            f"`{core_attention_attr}`."
        )
    core_attn = getattr(attention_layer, core_attention_attr)

    pre_handle = core_attn.register_forward_pre_hook(
        _make_pre_hook(fs, kv_arg_indices=kv_arg_indices)
    )
    post_handle = core_attn.register_forward_post_hook(_make_post_hook(fs))
    attention_layer._dha_pre_hook_handle = pre_handle
    attention_layer._dha_post_hook_handle = post_handle
    return fs


def install_all_fusion_hooks(
    transformer_layers: Sequence[nn.Layer],
    groupings: Sequence[Sequence[Sequence[int]]],
    *,
    attention_attr: str = "self_attn",
    core_attention_attr: str = "core_attention",
    kv_arg_indices: Tuple[int, int] = (1, 2),
    n_groups: int = 2,
    total_q_heads: int = 16,
    head_dim: int = 128,
    postmix_rank: int = 4,
    omega_init_per_layer: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[FusionState]:
    assert len(transformer_layers) == len(groupings), (
        f"#layers {len(transformer_layers)} != #groupings {len(groupings)}"
    )
    if omega_init_per_layer is not None:
        assert len(omega_init_per_layer) == len(transformer_layers), (
            f"#omega_init {len(omega_init_per_layer)} != #layers "
            f"{len(transformer_layers)}"
        )
    fusion_states: List[FusionState] = []
    for layer_idx, (block, groups) in enumerate(zip(transformer_layers, groupings)):
        if not hasattr(block, attention_attr):
            raise AttributeError(
                f"Transformer block {layer_idx} has no `{attention_attr}` submodule"
            )
        attn = getattr(block, attention_attr)
        n_src = sum(len(g) for g in groups)
        assignment = [-1] * n_src
        for gi, heads in enumerate(groups):
            for h in heads:
                assignment[h] = gi
        if -1 in assignment:
            raise ValueError(f"layer {layer_idx}: groups dont partition heads: {groups}")
        if omega_init_per_layer is not None:
            omk, omv = omega_init_per_layer[layer_idx]
        else:
            omk, omv = None, None
        fs = install_fusion_hooks(
            attn,
            head_assignment=assignment,
            core_attention_attr=core_attention_attr,
            kv_arg_indices=kv_arg_indices,
            n_groups=n_groups,
            total_q_heads=total_q_heads,
            head_dim=head_dim,
            postmix_rank=postmix_rank,
            omega_k_init=omk,
            omega_v_init=omv,
        )
        fusion_states.append(fs)
    return fusion_states


def collect_fusion_params(fusion_states: Sequence[FusionState]) -> List[paddle.Tensor]:
    params: List[paddle.Tensor] = []
    for fs in fusion_states:
        params.extend(fs.parameters())
    return params


def reset_caches(fusion_states: Sequence[FusionState]) -> None:
    for fs in fusion_states:
        fs.cached_k_pre_fusion = None
        fs.cached_v_pre_fusion = None
