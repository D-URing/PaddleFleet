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
Qwen model providers for VHA training.

Provides both GQA baseline and VHA variants as GPTModelProvider subclasses.
GQA uses standard gpt_builder, VHA uses vha_gpt_builder.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import paddle
import paddle.nn.functional as F

from paddleformers.transformers.gpt_provider import GPTModelProvider
from paddlefleet.gpt_builders import gpt_builder

from models.vha_builder import vha_gpt_builder

logger = logging.getLogger(__name__)


# =============================================================================
# GQA Baseline Providers
# =============================================================================

@dataclass
class QwenDenseGQAProvider(GPTModelProvider):
    """Base provider for Qwen3-style Dense GQA models."""

    normalization: str = "RMSNorm"
    hidden_act: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    attention_bias: bool = False
    use_qk_norm: bool = True
    seq_length: int = 4096
    max_position_embeddings: int = 40960
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    vocab_size: int = 151936
    tie_word_embeddings: Optional[bool] = True
    layernorm_epsilon: float = 1e-6
    autocast_dtype: paddle.dtype = paddle.bfloat16
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True

    attention_dropout: float = 0.0
    head_dim: int = 128

    position_embedding_type: str = "rope"
    rotary_base: float = 1000000.0
    rotary_percent: float = 1.0

    n_routed_experts: Optional[int] = None
    moe_grouped_gemm: bool = False

    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class Qwen3GQA_0p6B(QwenDenseGQAProvider):
    """Qwen3-0.6B: 28 layers, hidden=1024, 16 Q heads, 2 KV heads."""
    num_hidden_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 3072


@dataclass
class Qwen3GQA_1p7B(QwenDenseGQAProvider):
    """Qwen3-1.7B: 28 layers, hidden=2048, 16 Q heads, 2 KV heads."""
    num_hidden_layers: int = 28
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 6144


@dataclass
class Qwen3GQA_4B(QwenDenseGQAProvider):
    """Qwen3-4B: 36 layers, hidden=2560, 32 Q heads, 4 KV heads."""
    num_hidden_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 9216
    tie_word_embeddings: Optional[bool] = False


@dataclass
class Qwen3GQA_1p7B_SingleCard(QwenDenseGQAProvider):
    """Debug: small Qwen3-1.7B for single-card testing."""
    num_hidden_layers: int = 4
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 6144
    seq_length: int = 2048


# =============================================================================
# VHA Providers
# =============================================================================

@dataclass
class QwenVHAProvider(QwenDenseGQAProvider):
    """
    Provider for Qwen + VHA attention.

    Creates a GPT model with VHA attention layers using PaddleFleet native
    components. All transformer layers use VHASelfAttention.
    """

    # VHA-specific config
    vha_enable_premix: bool = True
    vha_enable_postmix: bool = True
    vha_postmix_rank: int = 4
    vha_premix_alpha_init: float = 5.0

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None):
        """Create GPT model with VHA attention layers."""
        pp_size = self.pipeline_model_parallel_size

        if hasattr(self, "rope_parameters") and self.rope_parameters:
            if "rope_type" in self.rope_parameters:
                if self.rope_parameters["rope_type"] != "default":
                    self.rope_type = self.rope_parameters["rope_type"]
            if "rope_theta" in self.rope_parameters:
                self.rope_theta = self.rope_parameters["rope_theta"]

        model = vha_gpt_builder(
            self,
            num_stages=pp_size,
            seg_method="layer:TransformerLayer|EmptyLayer",
            loss_fn=loss_fn,
        )
        return model


@dataclass
class Qwen3VHA_0p6B(QwenVHAProvider):
    """Qwen3-0.6B + VHA: 28 layers, hidden=1024, 16 Q heads, 2 KV heads."""
    num_hidden_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 3072
    vha_postmix_rank: int = 4


@dataclass
class Qwen3VHA_1p7B(QwenVHAProvider):
    """Qwen3-1.7B + VHA: 28 layers, hidden=2048, 16 Q heads, 2 KV heads."""
    num_hidden_layers: int = 28
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 6144
    vha_postmix_rank: int = 4


@dataclass
class Qwen3VHA_4B(QwenVHAProvider):
    """Qwen3-4B + VHA: 36 layers, hidden=2560, 32 Q heads, 4 KV heads."""
    num_hidden_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 9216
    tie_word_embeddings: Optional[bool] = False
    vha_postmix_rank: int = 4
