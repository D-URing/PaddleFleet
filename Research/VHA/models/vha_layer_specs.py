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
VHA layer specifications for PaddleFleet GPT model building.

Provides get_vha_layer_local_spec() which parallels get_gpt_layer_local_spec()
but uses VHASelfAttention instead of SelfAttention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.models.backends import LocalSpecProvider
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.paddle_norm import L2Norm
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)

from models.vha_attention import VHASelfAttention, VHASelfAttentionSublayersSpec

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


def get_vha_attention_spec(
    config: TransformerConfig,
    attn_mask_type: AttnMaskType = AttnMaskType.causal,
) -> LayerSpec:
    """Build VHA self-attention LayerSpec."""
    backend = LocalSpecProvider()

    use_qk_norm = getattr(config, "use_qk_norm", False)
    qk_l2_norm = getattr(config, "qk_l2_norm", False)

    if config.normalization == "RMSNorm":
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    use_triton_qk_norm = config.normalization == "RMSNorm" and getattr(
        config, "qk_norm_fusion", False
    )
    if use_triton_qk_norm:
        from paddlefleet.transformer.paddle_norm import WrappedRMSNormTriton
        qk_norm = WrappedRMSNormTriton

    return LayerSpec(
        layer=VHASelfAttention,
        extra_kwargs={"attn_mask_type": attn_mask_type},
        sublayers_spec=VHASelfAttentionSublayersSpec(
            qkv_proj=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            o_proj=backend.row_parallel_linear(),
            q_norm=(
                L2Norm if qk_l2_norm
                else (qk_norm if use_qk_norm else IdentityOp)
            ),
            k_norm=(
                L2Norm if qk_l2_norm
                else (qk_norm if use_qk_norm else IdentityOp)
            ),
        ),
    )


def get_vha_layer_local_spec(
    config: TransformerConfig | None = None,
    layer_number: int | None = 1,
    attn_mask_type: AttnMaskType = AttnMaskType.causal,
) -> LayerSpec:
    """Build a single transformer layer spec with VHA attention.

    Parallels get_gpt_layer_local_spec() but always uses VHASelfAttention.
    Dense MLP only (no MoE).
    """
    backend = LocalSpecProvider()

    if config.normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)

    # MLP
    if backend.fuse_layernorm_and_linear():
        up_gate_proj = backend.column_parallel_layer_norm_linear()
        assert up_gate_proj is not None
    else:
        up_gate_proj = backend.column_parallel_linear()

    mlp = LayerSpec(
        layer=MLP,
        sublayers_spec=MLPSublayersSpec(
            up_gate_proj=up_gate_proj,
            down_proj=backend.row_parallel_linear(),
            hidden_act=None,
        ),
    )

    self_attn_spec = get_vha_attention_spec(
        config=config,
        attn_mask_type=attn_mask_type,
    )

    return LayerSpec(
        layer=TransformerLayer,
        sublayers_spec=TransformerLayerSublayersSpec(
            input_layernorm=layer_norm,
            self_attn=self_attn_spec,
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            block_attn_res=IdentityOp,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attn.qkv_proj.layer_norm_",
                "post_attention_layernorm.": "mlp.up_gate_proj.layer_norm_",
            },
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
            "hidden_dropout_prob": config.hidden_dropout_prob
            if config is not None
            else None,
        },
    )
