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

"""Qwen model provider for UHA training."""

import json
import logging
import os
from dataclasses import dataclass, fields
from typing import Callable, Optional

import paddle
import paddle.nn.functional as F

from paddleformers.transformers.gpt_provider import GPTModelProvider
from paddlefleet.gpt_builders import gpt_builder

from models.uha_builder import uha_gpt_builder

logger = logging.getLogger(__name__)


@dataclass
class QwenBaseProvider(GPTModelProvider):
    """Base provider with Qwen3 defaults and UHA config fields."""

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

    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    num_hidden_layers: int = 28
    intermediate_size: int = 6144

    uha_num_value_heads: int = 1
    uha_value_head_dim: Optional[int] = 512
    uha_softmax_scale: Optional[float] = None
    uha_use_triton_kernel: bool = True

    loss_subbatch_sequence_length: int = 1024
    attn_type: str = "uha"

    def load_config(self, model_name_or_path: str):
        config_path = os.path.join(model_name_or_path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        field_names = {fld.name for fld in fields(self)}
        key_mapping = {"rms_norm_eps": "layernorm_epsilon"}
        act_mapping = {
            "silu": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
        }

        for k, v in cfg.items():
            mapped_k = key_mapping.get(k, k)
            if mapped_k in field_names:
                if mapped_k == "hidden_act" and isinstance(v, str):
                    v = act_mapping.get(v, v)
                setattr(self, mapped_k, v)

        logger.info(
            f"Loaded config from {config_path}: "
            f"attn_type={self.attn_type}, "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_attention_heads}, "
            f"num_kv_heads={self.num_key_value_heads}, "
            f"uha_value_heads={self.uha_num_value_heads}, "
            f"uha_value_head_dim={self.uha_value_head_dim}, "
            f"layers={self.num_hidden_layers}"
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None):
        pp_size = self.pipeline_model_parallel_size

        if hasattr(self, "rope_parameters") and self.rope_parameters:
            if "rope_type" in self.rope_parameters:
                if self.rope_parameters["rope_type"] != "default":
                    self.rope_type = self.rope_parameters["rope_type"]
            if "rope_theta" in self.rope_parameters:
                self.rope_theta = self.rope_parameters["rope_theta"]

        if self.attn_type == "uha":
            model = uha_gpt_builder(
                self,
                num_stages=pp_size,
                seg_method="layer:TransformerLayer|EmptyLayer",
                loss_fn=loss_fn,
            )
        else:
            model = gpt_builder(
                self,
                num_stages=pp_size,
                seg_method="layer:TransformerLayer|EmptyLayer",
                loss_fn=loss_fn,
            )

        return model


def create_provider(model_name_or_path: str) -> QwenBaseProvider:
    provider = QwenBaseProvider()
    provider.load_config(model_name_or_path)
    return provider
