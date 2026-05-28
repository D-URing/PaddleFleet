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

"""
UHA (Unequal-Head Attention) layer built on PaddleFleet components.

Q uses many small score heads, K uses fewer small score heads split from the
total K width, and V keeps one or more unsplit wide value heads. Example:
  - Q: 32 heads x 128 dim
  - K: 4 heads x 128 dim, each K head shared by 8 Q heads
  - V: 1 head x 512 dim, shared by all Q/K score groups
  - O projection maps the 512-dim UHA context back to hidden_size
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.utils import divide, get_pg_size

from kernels.uha_triton import uha_causal_attention_triton

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


@dataclass
class UHASelfAttentionSublayersSpec(SelfAttentionSublayersSpec):
    """Extends SelfAttentionSublayersSpec for UHA-specific projections."""
    pass


class UHACoreAttention(nn.Layer):
    """UHA core attention with Triton causal fast path and Paddle fallback."""

    def __init__(
        self,
        config: TransformerConfig,
        qk_heads_per_partition: int,
        v_heads_per_partition: int,
        qk_head_dim: int,
        v_head_dim: int,
    ):
        super().__init__()
        self.config = config
        self.qk_heads_per_partition = qk_heads_per_partition
        self.v_heads_per_partition = v_heads_per_partition
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.softmax_scale = getattr(config, "uha_softmax_scale", None)
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(qk_head_dim)
        self.use_triton_kernel = getattr(config, "uha_use_triton_kernel", True)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _expand_key_to_query_heads(self, key: Tensor, q_heads: int) -> Tensor:
        k_heads = key.shape[2]
        if q_heads == k_heads:
            return key
        if q_heads % k_heads != 0:
            raise ValueError(
                "UHA requires Q heads to be a multiple of K heads, got "
                f"q_heads={q_heads}, k_heads={k_heads}."
            )
        return key.repeat_interleave(q_heads // k_heads, axis=2)

    def _align_attention_probs_to_value_heads(self, attention_probs: Tensor) -> Tensor:
        q_heads = attention_probs.shape[1]
        v_heads = self.v_heads_per_partition

        if q_heads == v_heads:
            return attention_probs
        if q_heads % v_heads == 0:
            group_size = q_heads // v_heads
            attention_probs = attention_probs.reshape(
                [
                    attention_probs.shape[0],
                    v_heads,
                    group_size,
                    attention_probs.shape[2],
                    attention_probs.shape[3],
                ]
            )
            return attention_probs.mean(axis=2)
        raise ValueError(
            "UHA requires Q heads to be a multiple of V heads, got "
            f"q_heads={q_heads}, v_heads={v_heads}."
        )

    def _apply_attention_mask(
        self,
        attention_scores: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType | None,
    ) -> Tensor:
        if attention_mask is not None:
            if attention_mask.dtype == paddle.bool:
                min_value = paddle.full_like(attention_scores, -1e4)
                attention_scores = paddle.where(attention_mask, min_value, attention_scores)
            else:
                attention_scores = attention_scores + attention_mask.astype(attention_scores.dtype)
            return attention_scores

        if attn_mask_type == AttnMaskType.causal:
            q_len = attention_scores.shape[-2]
            k_len = attention_scores.shape[-1]
            causal_mask = paddle.triu(
                paddle.ones([q_len, k_len], dtype="bool"), diagonal=1
            ).reshape([1, 1, q_len, k_len])
            min_value = paddle.full_like(attention_scores, -1e4)
            attention_scores = paddle.where(causal_mask, min_value, attention_scores)
        return attention_scores

    def _forward_paddle(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType | None,
    ) -> Tensor:
        batch_size, q_len, q_heads, _ = query.shape
        k_len = key.shape[1]
        key = self._expand_key_to_query_heads(key, q_heads)

        query = query.transpose([0, 2, 1, 3]).reshape(
            [batch_size * q_heads, q_len, self.qk_head_dim]
        )
        key = key.transpose([0, 2, 3, 1]).reshape(
            [batch_size * q_heads, self.qk_head_dim, k_len]
        )

        attention_scores = paddle.bmm(query, key) * self.softmax_scale
        attention_scores = attention_scores.reshape([batch_size, q_heads, q_len, k_len])
        attention_scores = self._apply_attention_mask(
            attention_scores, attention_mask, attn_mask_type
        )

        attention_probs = paddle.nn.functional.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs)
        attention_probs = self._align_attention_probs_to_value_heads(attention_probs)

        v_heads = self.v_heads_per_partition
        value = value.transpose([0, 2, 1, 3]).reshape(
            [batch_size * v_heads, k_len, self.v_head_dim]
        )
        attention_probs = attention_probs.reshape([batch_size * v_heads, q_len, k_len])

        context = paddle.bmm(attention_probs, value)
        context = context.reshape([batch_size, v_heads, q_len, self.v_head_dim])
        context = context.transpose([0, 2, 1, 3]).contiguous()
        return context.reshape([batch_size, q_len, v_heads * self.v_head_dim])

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_startend_row_indices: Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Tensor | None = None,
        use_rr_flash_attention: bool = False,
    ) -> Tensor:
        if attention_bias is not None:
            raise ValueError("UHA attention does not support attention_bias yet.")
        if packed_seq_params is not None or attn_mask_startend_row_indices is not None:
            raise ValueError("UHA packed sequence/flashmask path needs a varlen Triton kernel.")

        can_use_triton = (
            self.use_triton_kernel
            and attention_mask is None
            and attn_mask_type == AttnMaskType.causal
            and query.dtype in (paddle.float16, paddle.bfloat16)
        )
        if can_use_triton:
            return uha_causal_attention_triton(
                query,
                key,
                value,
                softmax_scale=self.softmax_scale,
            )

        return self._forward_paddle(query, key, value, attention_mask, attn_mask_type)


class UHASelfAttention(SelfAttention):
    """UHA Self-Attention: Q/K small compute heads, V unequal wide heads."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: UHASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        tp_size = get_pg_size(self.pg_collection.tp)
        self.uha_qk_head_dim = self.hidden_size_per_attention_head
        self.uha_num_value_heads = getattr(config, "uha_num_value_heads", 1)
        self.uha_value_head_dim = getattr(config, "uha_value_head_dim", None)
        if self.uha_value_head_dim is None:
            self.uha_value_head_dim = divide(config.hidden_size, self.uha_num_value_heads)
        self.uha_value_heads_per_partition = divide(self.uha_num_value_heads, tp_size)
        self.uha_value_projection_size = self.uha_num_value_heads * self.uha_value_head_dim
        self.uha_value_projection_size_per_partition = (
            self.uha_value_heads_per_partition * self.uha_value_head_dim
        )
        self.uha_q_projection_size_per_partition = (
            self.num_attention_heads_per_partition * self.uha_qk_head_dim
        )
        self.uha_k_projection_size_per_partition = (
            self.num_query_groups_per_partition * self.uha_qk_head_dim
        )

        if self.num_attention_heads_per_partition % self.num_query_groups_per_partition != 0:
            raise ValueError(
                "UHA expects local Q heads to be a multiple of local K heads, got "
                f"q_heads={self.num_attention_heads_per_partition}, "
                f"k_heads={self.num_query_groups_per_partition}."
            )
        if self.num_attention_heads_per_partition % self.uha_value_heads_per_partition != 0:
            raise ValueError(
                "UHA expects local QK heads to be a multiple of local V heads, got "
                f"qk_heads={self.num_attention_heads_per_partition}, "
                f"v_heads={self.uha_value_heads_per_partition}."
            )

        self.qkv_proj = build_spec_layer(
            sublayers_spec.qkv_proj,
            self.config.hidden_size,
            self.query_projection_size
            + self.kv_projection_size
            + self.uha_value_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias or self.config.attention_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        if self.config.use_bias or self.config.attention_bias:
            raise ValueError(
                "UHA qkv split currently assumes qkv_proj returns only the projected tensor. "
                "Set use_bias=false and attention_bias=false."
            )

        self.core_attention = UHACoreAttention(
            config=config,
            qk_heads_per_partition=self.num_attention_heads_per_partition,
            v_heads_per_partition=self.uha_value_heads_per_partition,
            qk_head_dim=self.uha_qk_head_dim,
            v_head_dim=self.uha_value_head_dim,
        )

        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            self.uha_value_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

    def _get_uha_query_key_value_tensors(self, hidden_states: Tensor):
        mixed_qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = paddle.split(
            mixed_qkv,
            [
                self.uha_q_projection_size_per_partition,
                self.uha_k_projection_size_per_partition,
                self.uha_value_projection_size_per_partition,
            ],
            axis=-1,
        )
        query = query.reshape(
            [
                query.shape[0],
                query.shape[1],
                self.num_attention_heads_per_partition,
                self.uha_qk_head_dim,
            ]
        )
        key = key.reshape(
            [
                key.shape[0],
                key.shape[1],
                self.num_query_groups_per_partition,
                self.uha_qk_head_dim,
            ]
        )
        value = value.reshape(
            [
                value.shape[0],
                value.shape[1],
                self.uha_value_heads_per_partition,
                self.uha_value_head_dim,
            ]
        )
        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states: Tensor | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rope_freqs_cis: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Tensor | None = None,
        in_recompute: bool = False,
    ) -> tuple[Tensor, Tensor]:
        from paddle.distributed.fleet.utils import recompute
        from paddlefleet.models.common.embeddings import apply_rotary_pos_emb
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        if key_value_states is not None:
            raise ValueError("UHA self-attention does not support cross key_value_states.")
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self._get_uha_query_key_value_tensors(hidden_states)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if packed_seq_params is not None:
                cu_seqlens_q = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None
                    else packed_seq_params.cu_seqlens_q
                )
                cu_seqlens_kv = (
                    packed_seq_params.cu_seqlens_kv_padded
                    if packed_seq_params.cu_seqlens_kv_padded is not None
                    else packed_seq_params.cu_seqlens_kv
                )
                total_seqlen_q = packed_seq_params.total_seqlen_q
                total_seqlen_kv = packed_seq_params.total_seqlen_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
                total_seqlen_q = total_seqlen_kv = None

            if (
                self.config.apply_rope_fusion
                and not self.config.high_precision_rope
                and q_pos_emb is not None
                and k_pos_emb is not None
            ):
                query, key, _ = apply_rotary_pos_emb(
                    (query, key),
                    None,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    position_ids=position_ids,
                    mscale=None,
                    cp_group=self.pg_collection.cp,
                )
            else:
                if q_pos_emb is not None:
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        None,
                        None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        total_seq_len=total_seqlen_q,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        None,
                        None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        total_seq_len=total_seqlen_kv,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )

        if self.config.sequence_parallel:
            query = query.transpose([1, 0, 2, 3]).contiguous()
            key = key.transpose([1, 0, 2, 3]).contiguous()
            value = value.transpose([1, 0, 2, 3]).contiguous()

        if self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                self.core_attention,
                query,
                key,
                value,
                attention_mask.clone() if attention_mask is not None else None,
                attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None
                else None,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_startend_row_indices,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention and in_recompute,
            )

        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        output, bias = self.o_proj(core_attn_out)
        return output, bias

    def uha_param_groups(self):
        return {
            "qk_heads_per_partition": self.num_attention_heads_per_partition,
            "v_heads_per_partition": self.uha_value_heads_per_partition,
            "qk_head_dim": self.uha_qk_head_dim,
            "v_head_dim": self.uha_value_head_dim,
        }
