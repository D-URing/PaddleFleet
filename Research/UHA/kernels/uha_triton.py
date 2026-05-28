# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton forward kernel for UHA causal attention.

The kernel computes attention scores with many Q heads, fewer K heads, and one
or more unsplit wide V heads. For example, 32 Q heads attend to 4 K heads
(repeating each K head for 8 Q heads), then all 32 probability maps are grouped
and averaged into a single 512-dim V aggregation when HV=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import paddle

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - handled by Python fallback path.
    triton = None
    tl = None


@dataclass(frozen=True)
class UHATritonConfig:
    block_m: int = 16
    block_n: int = 64
    block_dqk: int = 128
    block_dv: int = 128
    num_warps: int = 4
    num_stages: int = 3


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def get_uha_triton_config(qk_head_dim: int, v_head_dim: int) -> UHATritonConfig:
    return UHATritonConfig(
        block_dqk=_next_power_of_2(qk_head_dim),
        block_dv=min(_next_power_of_2(v_head_dim), 128),
        num_warps=4,
        num_stages=3 if qk_head_dim <= 128 else 2,
    )


if triton is not None:

    @triton.jit
    def _uha_causal_forward_kernel(
        Q,
        K,
        V,
        O,
        B: tl.constexpr,
        S: tl.constexpr,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        HV: tl.constexpr,
        DQK: tl.constexpr,
        DV: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DQK: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_dv = tl.program_id(1)
        pid_b = tl.program_id(2)
        pid_vh = tl.program_id(3)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_dqk = tl.arange(0, BLOCK_DQK)
        offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

        acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        v_head_acc_scale = 1.0 / GROUP_SIZE

        for group_offset in range(0, GROUP_SIZE):
            q_head = pid_vh * GROUP_SIZE + group_offset
            k_head = q_head // (HQ // HK)
            m_i = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc_i = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

            q = tl.load(
                Q
                + ((pid_b * S + offs_m[:, None]) * HQ + q_head) * DQK
                + offs_dqk[None, :],
                mask=(offs_m[:, None] < S) & (offs_dqk[None, :] < DQK),
                other=0.0,
            )

            for start_n in range(0, S, BLOCK_N):
                cols = start_n + offs_n
                k = tl.load(
                    K
                    + ((pid_b * S + cols[None, :]) * HK + k_head) * DQK
                    + offs_dqk[:, None],
                    mask=(cols[None, :] < S) & (offs_dqk[:, None] < DQK),
                    other=0.0,
                )
                scores = tl.dot(q, k) * SCALE
                causal_mask = cols[None, :] <= offs_m[:, None]
                scores = tl.where(causal_mask & (offs_m[:, None] < S), scores, -float("inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_new[:, None])
                alpha = tl.exp(m_i - m_new)
                l_new = l_i * alpha + tl.sum(p, axis=1)

                v = tl.load(
                    V
                    + ((pid_b * S + cols[:, None]) * HV + pid_vh) * DV
                    + offs_dv[None, :],
                    mask=(cols[:, None] < S) & (offs_dv[None, :] < DV),
                    other=0.0,
                )
                acc_i = acc_i * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
                l_i = l_new

            acc_i = acc_i / l_i[:, None]
            acc += acc_i * v_head_acc_scale

        tl.store(
            O + ((pid_b * S + offs_m[:, None]) * HV + pid_vh) * DV + offs_dv[None, :],
            acc,
            mask=(offs_m[:, None] < S) & (offs_dv[None, :] < DV),
        )


def _validate_uha_inputs(query: paddle.Tensor, key: paddle.Tensor, value: paddle.Tensor):
    if triton is None:
        raise ImportError("triton is required for uha_causal_attention_triton")
    if query.dtype not in (paddle.float16, paddle.bfloat16):
        raise ValueError(f"UHA Triton kernel supports fp16/bf16 only, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key and value must have the same dtype")

    batch_size, seq_len, q_heads, qk_head_dim = query.shape
    k_heads = key.shape[2]
    v_heads = value.shape[2]
    v_head_dim = value.shape[3]

    if key.shape[0] != batch_size or key.shape[1] != seq_len or key.shape[3] != qk_head_dim:
        raise ValueError(f"key shape is incompatible with query, got {key.shape} vs {query.shape}")
    if value.shape[0] != batch_size or value.shape[1] != seq_len:
        raise ValueError("value must share batch and sequence dimensions with query")
    if q_heads % k_heads != 0:
        raise ValueError(
            "UHA Triton kernel expects q_heads to be a multiple of k_heads, "
            f"got q_heads={q_heads}, k_heads={k_heads}"
        )
    if q_heads % v_heads != 0:
        raise ValueError(
            "UHA Triton kernel expects q_heads to be a multiple of v_heads, "
            f"got q_heads={q_heads}, v_heads={v_heads}"
        )
    return batch_size, seq_len, q_heads, k_heads, qk_head_dim, v_heads, v_head_dim


def _uha_causal_attention_triton_forward(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    softmax_scale: float,
    config: UHATritonConfig | None = None,
) -> paddle.Tensor:
    batch_size, seq_len, q_heads, k_heads, qk_head_dim, v_heads, v_head_dim = _validate_uha_inputs(
        query, key, value
    )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = paddle.empty([batch_size, seq_len, v_heads, v_head_dim], dtype=value.dtype)

    cfg = config or get_uha_triton_config(qk_head_dim, v_head_dim)
    group_size = q_heads // v_heads
    grid = (
        triton.cdiv(seq_len, cfg.block_m),
        triton.cdiv(v_head_dim, cfg.block_dv),
        batch_size,
        v_heads,
    )

    _uha_causal_forward_kernel[grid](
        query,
        key,
        value,
        output,
        batch_size,
        seq_len,
        q_heads,
        k_heads,
        v_heads,
        qk_head_dim,
        v_head_dim,
        softmax_scale,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_DQK=cfg.block_dqk,
        BLOCK_DV=cfg.block_dv,
        GROUP_SIZE=group_size,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return output.reshape([batch_size, seq_len, v_heads * v_head_dim])


def _uha_causal_attention_paddle_backward(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    grad_output: paddle.Tensor,
    softmax_scale: float,
):
    batch_size, seq_len, q_heads, qk_head_dim = query.shape
    k_heads = key.shape[2]
    v_heads = value.shape[2]
    v_head_dim = value.shape[3]
    q_per_k = q_heads // k_heads
    q_per_v = q_heads // v_heads

    query_t = query.transpose([0, 2, 1, 3]).astype("float32")
    key_t = key.transpose([0, 2, 1, 3]).repeat_interleave(q_per_k, axis=1).astype("float32")
    value_t = value.transpose([0, 2, 1, 3]).astype("float32")

    scores = paddle.matmul(query_t, key_t, transpose_y=True) * softmax_scale
    causal_mask = paddle.triu(
        paddle.ones([seq_len, seq_len], dtype="bool"), diagonal=1
    ).reshape([1, 1, seq_len, seq_len])
    scores = paddle.where(causal_mask, paddle.full_like(scores, -1e9), scores)
    probs = paddle.nn.functional.softmax(scores, axis=-1)

    value_per_q = value_t.repeat_interleave(q_per_v, axis=1)
    grad_context = grad_output.reshape([batch_size, seq_len, v_heads, v_head_dim])
    grad_context = grad_context.transpose([0, 2, 1, 3]).repeat_interleave(
        q_per_v, axis=1
    ) / q_per_v
    grad_context = grad_context.astype("float32")

    grad_value_per_q = paddle.matmul(probs, grad_context, transpose_x=True)
    grad_value = grad_value_per_q.reshape(
        [batch_size, v_heads, q_per_v, seq_len, v_head_dim]
    ).sum(axis=2)

    grad_probs = paddle.matmul(grad_context, value_per_q, transpose_y=True)
    grad_scores = probs * (grad_probs - (grad_probs * probs).sum(axis=-1, keepdim=True))
    grad_scores = paddle.where(causal_mask, paddle.zeros_like(grad_scores), grad_scores)

    grad_query = paddle.matmul(grad_scores, key_t) * softmax_scale
    grad_key_per_q = paddle.matmul(grad_scores, query_t, transpose_x=True) * softmax_scale
    grad_key = grad_key_per_q.reshape(
        [batch_size, k_heads, q_per_k, seq_len, qk_head_dim]
    ).sum(axis=2)

    grad_query = grad_query.transpose([0, 2, 1, 3]).astype(query.dtype)
    grad_key = grad_key.transpose([0, 2, 1, 3]).astype(key.dtype)
    grad_value = grad_value.transpose([0, 2, 1, 3]).astype(value.dtype)
    return grad_query, grad_key, grad_value


class UHACausalAttentionTriton(paddle.autograd.PyLayer):
    """Triton forward with Paddle autograd-compatible analytical backward."""

    @staticmethod
    def forward(ctx, query, key, value, softmax_scale):
        scale = float(softmax_scale)
        output = _uha_causal_attention_triton_forward(query, key, value, scale)
        ctx.save_for_backward(query, key, value)
        ctx.softmax_scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensor()
        grad_query, grad_key, grad_value = _uha_causal_attention_paddle_backward(
            query, key, value, grad_output, ctx.softmax_scale
        )
        return grad_query, grad_key, grad_value, None


def uha_causal_attention_triton(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    softmax_scale: float | None = None,
    config: UHATritonConfig | None = None,
) -> paddle.Tensor:
    """Run UHA causal attention with Triton forward and backward support.

    Args:
        query: [B, S, H_q, D_qk]
        key: [B, S, H_k, D_qk]
        value: [B, S, H_v, D_v]

    Returns:
        [B, S, H_v * D_v]
    """
    qk_head_dim = query.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qk_head_dim)
    if config is not None:
        return _uha_causal_attention_triton_forward(query, key, value, scale, config)
    return UHACausalAttentionTriton.apply(query, key, value, scale)
