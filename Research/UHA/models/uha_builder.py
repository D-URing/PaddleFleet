"""
UHA GPT model builder.

Builds a GPT model with UHA attention layers.
"""

from __future__ import annotations

from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.models.backends import LocalSpecProvider
from paddlefleet.models.common.empty_layer import EmptyLayer
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.paddle_norm import L2Norm
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)

from models.uha_attention import UHASelfAttention, UHASelfAttentionSublayersSpec


def _get_uha_layer_spec(config, layer_number=1, attn_mask_type=AttnMaskType.causal):
    """Build a single transformer layer spec with UHA attention."""
    backend = LocalSpecProvider()

    rms = config.normalization == "RMSNorm"
    layer_norm = backend.layer_norm(rms_norm=rms, for_qk=False)

    use_qk_norm = getattr(config, "use_qk_norm", False)
    qk_l2_norm = getattr(config, "qk_l2_norm", False)
    qk_norm = backend.layer_norm(rms_norm=rms, for_qk=True)

    if rms and getattr(config, "qk_norm_fusion", False):
        from paddlefleet.transformer.paddle_norm import WrappedRMSNormTriton
        qk_norm = WrappedRMSNormTriton

    def _pick_qk_norm():
        if qk_l2_norm:
            return L2Norm
        return qk_norm if use_qk_norm else IdentityOp

    self_attn_spec = LayerSpec(
        layer=UHASelfAttention,
        extra_kwargs={"attn_mask_type": attn_mask_type},
        sublayers_spec=UHASelfAttentionSublayersSpec(
            qkv_proj=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            o_proj=backend.row_parallel_linear(),
            q_norm=_pick_qk_norm(),
            k_norm=_pick_qk_norm(),
        ),
    )

    if backend.fuse_layernorm_and_linear():
        up_gate_proj = backend.column_parallel_layer_norm_linear()
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
            "hidden_dropout_prob": config.hidden_dropout_prob if config else None,
        },
    )


def uha_gpt_builder(config, **kwargs):
    """Build a GPT model with UHA attention layers."""
    print("building UHA GPT model ...")

    transformer_layers_spec = [
        _get_uha_layer_spec(
            config=config,
            layer_number=i + config.num_empty_layers_add_in_head,
        )
        for i in range(config.num_hidden_layers)
    ]

    head_empty_layers_spec = [
        LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        for _ in range(config.num_empty_layers_add_in_head)
    ]
    tail_empty_layers_spec = [
        LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        for _ in range(config.num_empty_layers_add_in_tail)
    ]

    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=head_empty_layers_spec,
        transformer_layers_spec=transformer_layers_spec,
        tail_empty_layers_spec=tail_empty_layers_spec,
        mtp_layers_spec=None,
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
