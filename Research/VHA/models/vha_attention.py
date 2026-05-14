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
  - Premix: per-KV-group rotation matrices [H_k, d, d] applied to Q after QKV projection
  - Postmix: low-rank (I + VU^T) cross-head mixing applied to attention output

Uses PaddleFleet's ColumnParallelLinear (QKV), RowParallelLinear (O),
and DotProductAttention (core attention) — fully compatible with TP/PP/SP.
"""

from __future__ import annotations

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
        2. QK norm — inherited
        3. **Premix**: per-KV-group rotation on Q heads (VHA-specific)
        4. RoPE — inherited
        5. Core attention (DotProductAttention) — inherited
        6. **Postmix**: low-rank cross-head mixing on attention output (VHA-specific)
        7. Output projection (RowParallelLinear) — inherited

    VHA config fields (read from TransformerConfig):
        - vha_enable_premix: bool (default True)
        - vha_enable_postmix: bool (default True)
        - vha_postmix_rank: int (default 4)
        - vha_premix_alpha_init: float (default 5.0)
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
        premix_alpha_init = getattr(config, "vha_premix_alpha_init", 5.0)

        # Head counts (per-partition for TP)
        H_k_local = self.num_query_groups_per_partition
        H_q_local = self.num_attention_heads_per_partition
        d = self.hidden_size_per_attention_head

        # Premix: rotation matrices [H_k_local, d, d] with alpha gating
        if self.vha_enable_premix:
            self.vha_premix_weight = self.create_parameter(
                shape=[H_k_local, d, d],
                default_initializer=nn.initializer.Assign(
                    paddle.eye(d).unsqueeze(0).expand([H_k_local, d, d])
                ),
            )
            self.vha_premix_alpha = self.create_parameter(
                shape=[H_k_local],
                default_initializer=nn.initializer.Constant(premix_alpha_init),
            )

        # Postmix: low-rank (I + VU^T) on attention output
        if self.vha_enable_postmix:
            r = self.vha_postmix_rank
            self.vha_postmix_U = self.create_parameter(
                shape=[H_q_local, r],
                default_initializer=nn.initializer.Normal(mean=0.0, std=0.01),
            )
            self.vha_postmix_V = self.create_parameter(
                shape=[H_q_local, r],
                default_initializer=nn.initializer.Constant(0.0),
            )

    def _apply_premix(self, query: Tensor) -> Tensor:
        """
        Apply premix rotation to query tensor.

        Args:
            query: [b, sq, np, hn] where np = H_q_local, hn = head_dim

        Returns:
            query with premix applied: same shape
        """
        if not self.vha_enable_premix:
            return query

        H_k_local = self.num_query_groups_per_partition
        H_q_local = self.num_attention_heads_per_partition
        heads_per_group = H_q_local // H_k_local
        d = self.hidden_size_per_attention_head

        # Effective premix: W_eff = I + sigmoid(alpha) * (W - I)
        alpha = paddle.sigmoid(self.vha_premix_alpha)  # [H_k_local]
        identity = paddle.eye(d).unsqueeze(0)  # [1, d, d]
        w_eff = identity + alpha.reshape([-1, 1, 1]) * (self.vha_premix_weight - identity)
        # w_eff: [H_k_local, d, d]

        # query: [b, sq, np, hn] -> [b, sq, H_k_local, heads_per_group, d]
        b, sq = query.shape[0], query.shape[1]
        q_grouped = query.reshape([b, sq, H_k_local, heads_per_group, d])

        # Apply rotation per KV group
        q_rotated = paddle.einsum("bsghd,gde->bsghe", q_grouped, w_eff)

        return q_rotated.reshape([b, sq, H_q_local, d])

    def _apply_postmix(self, attn_out: Tensor) -> Tensor:
        """
        Apply postmix low-rank cross-head mixing to attention output.

        Args:
            attn_out: [b, sq, h] where h = H_q_local * head_dim

        Returns:
            attn_out with postmix applied: same shape
        """
        if not self.vha_enable_postmix:
            return attn_out

        H_q_local = self.num_attention_heads_per_partition
        d = self.hidden_size_per_attention_head
        b, sq, _ = attn_out.shape

        # Reshape to [b, sq, H_q_local, d]
        x = attn_out.reshape([b, sq, H_q_local, d])

        # Low-rank mixing: delta = (x @ U) @ V^T in head dimension
        Z = paddle.einsum("bshd,hr->bsrd", x, self.vha_postmix_U)
        delta = paddle.einsum("bsrd,hr->bshd", Z, self.vha_postmix_V)

        x = x + delta
        return x.reshape([b, sq, H_q_local * d])

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
        VHA forward: QKV -> premix on Q -> RoPE -> core attention -> postmix -> O proj.
        """
        from paddle.distributed.fleet.utils import recompute
        from paddlefleet.models.common.embeddings import apply_rotary_pos_emb
        from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        no_rope = False
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # QKV projection
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=True
        )
        attn_mask_type = self.attn_mask_type

        if len(qkv_output) == 4:
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output
            gate = None

        # Premix on Q
        query = self._apply_premix(query)

        # RoPE
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

        # Postmix
        core_attn_out = self._apply_postmix(core_attn_out)

        # Output projection
        output, bias = self.o_proj(core_attn_out)

        return output, bias

    def vha_param_groups(self):
        """Return VHA-specific parameter groups for optimizer configuration."""
        groups = {"premix": [], "postmix": []}
        if self.vha_enable_premix:
            groups["premix"] = [self.vha_premix_weight, self.vha_premix_alpha]
        if self.vha_enable_postmix:
            groups["postmix"] = [self.vha_postmix_U, self.vha_postmix_V]
        return groups
