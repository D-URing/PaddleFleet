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
VHA Self-Attention layer built on PaddleFleet native components.

Extends PaddleFleet's SelfAttention with:
  - Premix: W[H_k, d, d] expands Q from H_q heads to H_k*H_q virtual heads
  - Postmix: low-rank UV cross-head mixing on expanded attention output

Uses PaddleFleet's ColumnParallelLinear (QKV), RowParallelLinear (O),
and DotProductAttention (core attention) — fully compatible with TP/PP/SP.

Architecture:
  num_attention_heads (config) = H_q = pre-expansion Q head count
  num_key_value_heads (config) = H_k = KV head count
  total_heads = H_k * H_q = expanded virtual head count after premix
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.utils import divide, get_pg_size

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


@dataclass
class VHASelfAttentionSublayersSpec(SelfAttentionSublayersSpec):
    """Extends SelfAttentionSublayersSpec — same components, VHA adds its own params."""
    pass


class VHASelfAttention(SelfAttention):
    """
    VHA Self-Attention: extends PaddleFleet SelfAttention with premix and postmix.

    Architecture:
        1. QKV projection (fused, ColumnParallelLinear) — inherited
           Q: [B, T, H_q_local, d], K: [B, T, H_k_local, d], V: [B, T, H_k_local, d]
        2. QK norm — inherited
        3. **Premix**: W[H_k, d, d] expands Q from H_q to H_k*H_q virtual heads
           Q: [B, T, H_q_local, d] -> [B, T, H_k_local*H_q_local, d]
        4. RoPE — on expanded Q and original K
        5. Core attention (DotProductAttention) — K/V internally repeated by factor H_q
        6. **Postmix**: low-rank UV cross-head mixing on H_k*H_q expanded output
        7. Output projection (RowParallelLinear) — input dim = H_k*H_q*d

    VHA config fields (read from TransformerConfig):
        - vha_enable_premix: bool (default True)
        - vha_enable_postmix: bool (default True)
        - vha_postmix_rank: int (default 4)
        - vha_premix_init_alpha: float (default 0.1)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: VHASelfAttentionSublayersSpec,
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

        # VHA config
        self.vha_enable_premix = getattr(config, "vha_enable_premix", True)
        self.vha_enable_postmix = getattr(config, "vha_enable_postmix", True)
        self.vha_postmix_rank = getattr(config, "vha_postmix_rank", 4)
        self.vha_premix_init_alpha = getattr(config, "vha_premix_init_alpha", 0.1)

        # Head counts (per-partition for TP)
        H_k_local = self.num_query_groups_per_partition   # H_k / tp
        H_q_local = self.num_attention_heads_per_partition  # H_q / tp
        d = self.hidden_size_per_attention_head
        total_heads_local = H_k_local * H_q_local  # expanded virtual heads per partition

        self.total_heads_local = total_heads_local

        # --- Override core_attention head counts for correct K/V expansion ---
        # After premix, Q has total_heads_local heads. K/V have H_k_local heads.
        # We need core_attention to repeat K/V by factor total_heads_local / H_k_local = H_q_local.
        # Original config gives repeat factor = H_q_local / H_k_local (wrong for expanded Q).
        # Override to: num_attention_heads_per_partition = total_heads_local, keeping
        # num_query_groups_per_partition = H_k_local (unchanged).
        self.core_attention.num_attention_heads_per_partition = total_heads_local
        # num_query_groups_per_partition stays H_k_local (already correct from config)

        # --- Rebuild o_proj with correct input dimension ---
        # After expansion, attention output has total_heads * d dimensions.
        # Original o_proj expects H_q * d. We need H_k * H_q * d.
        total_query_projection_size = (
            config.num_key_value_heads * config.num_attention_heads * d
        )
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            total_query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )

        # --- Premix: W[H_k_local, d, d] ---
        # Initialized as I + alpha/sqrt(d) * randn (LD mode)
        if self.vha_enable_premix:
            alpha = self.vha_premix_init_alpha
            I = paddle.eye(d)
            init_mats = paddle.stack([
                I + paddle.randn([d, d]) * (alpha / math.sqrt(d))
                for _ in range(H_k_local)
            ])
            self.vha_premix_weight = self.create_parameter(
                shape=[H_k_local, d, d],
                default_initializer=nn.initializer.Assign(init_mats),
            )

        # --- Postmix: UV low-rank on expanded heads ---
        # U, V: [total_heads_local, r]
        if self.vha_enable_postmix:
            r = self.vha_postmix_rank
            self.vha_postmix_U = self.create_parameter(
                shape=[total_heads_local, r],
                default_initializer=nn.initializer.Normal(mean=0.0, std=0.01),
            )
            self.vha_postmix_V = self.create_parameter(
                shape=[total_heads_local, r],
                default_initializer=nn.initializer.Constant(0.0),
            )

    def _apply_premix(self, query: Tensor) -> Tensor:
        """
        Expand Q from H_q heads to H_k*H_q virtual heads via premix weight.

        Args:
            query: [b, sq, H_q_local, d]

        Returns:
            expanded query: [b, sq, H_k_local * H_q_local, d]

        Operation:
            einsum("bthd,kde->btkhe", Q[B,T,H_q,d], W[H_k,d,d])
            -> [B, T, H_k, H_q, d] -> reshape [B, T, H_k*H_q, d]
        """
        if not self.vha_enable_premix:
            return query

        # query: [b, sq, H_q_local, d]
        # vha_premix_weight: [H_k_local, d, d]
        # Result: [b, sq, H_k_local, H_q_local, d]
        q_expanded = paddle.einsum(
            "bthd,kde->btkhe", query, self.vha_premix_weight
        )
        # Reshape to [b, sq, H_k_local * H_q_local, d]
        b, sq = query.shape[0], query.shape[1]
        return q_expanded.reshape([b, sq, self.total_heads_local, self.hidden_size_per_attention_head])

    def _apply_postmix(self, attn_out: Tensor) -> Tensor:
        """
        Apply postmix low-rank UV cross-head mixing on expanded attention output.

        Args:
            attn_out: [b, sq, total_heads_local * d] (flattened)

        Returns:
            attn_out with postmix applied: same shape
        """
        if not self.vha_enable_postmix:
            return attn_out

        d = self.hidden_size_per_attention_head
        b, sq, _ = attn_out.shape

        # Reshape to [b, sq, total_heads_local, d]
        x = attn_out.reshape([b, sq, self.total_heads_local, d])

        # UV mixing: delta = V @ (U^T @ x) in head dimension
        # Z = einsum("bthd,hr->btrd", x, U)  -> [b, sq, r, d]
        # delta = einsum("btrd,hr->bthd", Z, V) -> [b, sq, total_heads_local, d]
        Z = paddle.einsum("bthd,hr->btrd", x, self.vha_postmix_U)
        delta = paddle.einsum("btrd,hr->bthd", Z, self.vha_postmix_V)

        x = x + delta
        return x.reshape([b, sq, self.total_heads_local * d])

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
        """
        VHA forward: QKV -> premix (expand Q) -> RoPE -> core attention -> postmix -> O proj.
        """
        from paddle.distributed.fleet.utils import recompute
        from paddlefleet.models.common.embeddings import apply_rotary_pos_emb
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # QKV projection: Q[B,T,H_q_local,d], K[B,T,H_k_local,d], V[B,T,H_k_local,d]
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=True
        )
        attn_mask_type = self.attn_mask_type

        if len(qkv_output) == 4:
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output
            gate = None

        # Premix: expand Q from [B,T,H_q_local,d] to [B,T,H_k_local*H_q_local,d]
        query = self._apply_premix(query)

        # RoPE (applied on expanded Q and original K)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
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
                    (query, key), None,
                    rotary_pos_cos, rotary_pos_sin,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    position_ids=position_ids,
                    mscale=None,
                    cp_group=self.pg_collection.cp,
                )
            else:
                if q_pos_emb is not None:
                    query = apply_rotary_pos_emb(
                        query, q_pos_emb, None, None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        total_seq_len=total_seqlen_q,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key, k_pos_emb, None, None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        total_seq_len=total_seqlen_kv,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )

        # Core attention
        # Q: [B, T, total_heads_local, d], K: [B, T, H_k_local, d], V: [B, T, H_k_local, d]
        # core_attention internally repeats K/V by factor total_heads_local / H_k_local = H_q_local
        if self.config.sequence_parallel:
            query = query.transpose([1, 0, 2, 3]).contiguous()
            key = key.transpose([1, 0, 2, 3]).contiguous()
            value = value.transpose([1, 0, 2, 3]).contiguous()

        if self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                self.core_attention,
                query, key, value,
                attention_mask.clone() if attention_mask is not None else None,
                attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None else None,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention,
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value,
                attention_mask,
                attn_mask_startend_row_indices,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention and in_recompute,
            )

        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        if gate is not None:
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)

        # Postmix on expanded heads
        core_attn_out = self._apply_postmix(core_attn_out)

        # Output projection (input dim = total_heads * d = H_k * H_q * d)
        output, bias = self.o_proj(core_attn_out)

        return output, bias

    def vha_param_groups(self):
        """Return VHA-specific parameter groups for optimizer configuration."""
        groups = {"premix": [], "postmix": []}
        if self.vha_enable_premix:
            groups["premix"] = [self.vha_premix_weight]
        if self.vha_enable_postmix:
            groups["postmix"] = [self.vha_postmix_U, self.vha_postmix_V]
        return groups
