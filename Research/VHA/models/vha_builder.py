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
VHA GPT model builder.

Parallels paddlefleet.gpt_builders.gpt_builder() but uses VHA layer specs
for transformer layers. GQA baseline uses the standard gpt_builder.
"""

from __future__ import annotations

from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.models.common.empty_layer import EmptyLayer
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec

from models.vha_layer_specs import get_vha_layer_local_spec


def vha_gpt_builder(config, **kwargs):
    """Build a GPT model with VHA attention layers.

    Same structure as paddlefleet.gpt_builders.gpt_builder() but uses
    get_vha_layer_local_spec for transformer layers.

    Args:
        config: TransformerConfig with VHA fields (vha_enable_premix, etc.)
        **kwargs: Additional kwargs passed to build_spec_layer.

    Returns:
        GPTModel with VHA attention in all transformer layers.
    """
    print("building VHA GPT model ...")

    # Build VHA transformer layer specs
    transformer_layers_spec = []
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        transformer_layers_spec.append(
            get_vha_layer_local_spec(
                config=config,
                layer_number=real_layer_number,
            )
        )

    mtp_layers_spec = None

    # Empty layers for pipeline parallelism
    head_empty_layers_spec = []
    for i in range(config.num_empty_layers_add_in_head):
        head_empty_layers_spec.append(
            LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        )

    tail_empty_layers_spec = []
    for i in range(config.num_empty_layers_add_in_tail):
        tail_empty_layers_spec.append(
            LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        )

    # Build complete GPT spec
    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=head_empty_layers_spec,
        transformer_layers_spec=transformer_layers_spec,
        tail_empty_layers_spec=tail_empty_layers_spec,
        mtp_layers_spec=mtp_layers_spec,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )

    loss_fn = kwargs.pop("loss_fn", None)

    return build_spec_layer(
        gpt_spec,
        loss_fn=LanguageLoss(config) if not loss_fn else loss_fn,
        **kwargs,
    )
