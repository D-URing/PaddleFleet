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
VHA warm-up utilities.

Handles:
  1. Loading GQA checkpoint into VHA model (init_vha_from_gqa_checkpoint)
  2. Per-layer VHA parameter freeze/unfreeze scheduling (VHAWarmupManager)
  3. VHA-specific optimizer parameter groups with per-group LR
"""

from __future__ import annotations

import logging
import os

import paddle

from models.vha_attention import VHASelfAttention

logger = logging.getLogger(__name__)


def find_vha_attention_layers(model) -> list:
    """Find all VHASelfAttention layers in the model."""
    vha_layers = []
    for sublayer in model.sublayers():
        if isinstance(sublayer, VHASelfAttention):
            vha_layers.append(sublayer)
    return vha_layers


def init_vha_from_gqa_checkpoint(
    model,
    gqa_checkpoint_dir: str,
    alpha_init: float = 5.0,
):
    """
    Initialize VHA model from a GQA checkpoint.

    Loads the GQA checkpoint state dict, copies all matching weights
    (embedding, MLP, layer norm, QKV, O projections), and initializes
    VHA-specific parameters (premix, postmix) to identity-equivalent values.

    Args:
        model: VHA GPT model (built with vha_gpt_builder)
        gqa_checkpoint_dir: Path to GQA checkpoint directory
        alpha_init: Initial value for premix alpha (sigmoid(5)≈0.993)
    """
    # Find checkpoint file
    ckpt_path = None
    for fname in ["model_state.pdparams", "model.pdparams"]:
        candidate = os.path.join(gqa_checkpoint_dir, fname)
        if os.path.exists(candidate):
            ckpt_path = candidate
            break

    if ckpt_path is None:
        candidates = [
            f for f in os.listdir(gqa_checkpoint_dir)
            if f.endswith(".pdparams")
        ]
        if candidates:
            ckpt_path = os.path.join(gqa_checkpoint_dir, candidates[0])

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {gqa_checkpoint_dir}. "
            "Expected model_state.pdparams or *.pdparams file."
        )

    logger.info(f"Loading GQA checkpoint from: {ckpt_path}")
    gqa_state = paddle.load(ckpt_path)

    # Load matching weights
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in gqa_state.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in model)")

    model.set_state_dict(model_state)
    logger.info(f"Loaded {len(loaded_keys)} parameters from GQA checkpoint")
    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} parameters (VHA-specific or mismatched)")

    # Initialize VHA-specific parameters
    vha_layers = find_vha_attention_layers(model)
    logger.info(f"Initializing VHA params for {len(vha_layers)} attention layers")

    for layer in vha_layers:
        if layer.vha_enable_premix:
            d = layer.hidden_size_per_attention_head
            H_k_local = layer.num_query_groups_per_partition
            layer.vha_premix_weight.set_value(
                paddle.eye(d).unsqueeze(0).expand([H_k_local, d, d])
            )
            layer.vha_premix_alpha.set_value(
                paddle.full([H_k_local], alpha_init)
            )

        if layer.vha_enable_postmix:
            layer.vha_postmix_V.set_value(
                paddle.zeros_like(layer.vha_postmix_V)
            )
            layer.vha_postmix_U.set_value(
                paddle.normal(mean=0.0, std=0.01, shape=layer.vha_postmix_U.shape)
            )

    logger.info("VHA initialization from GQA checkpoint complete")


class VHAWarmupManager:
    """
    Manages VHA warm-up training phases.

    Phases:
        0 (stabilize): VHA params (premix/postmix) frozen, only backbone trains.
        1 (activate):  All parameters unfrozen, full training.

    Usage:
        warmup_mgr = VHAWarmupManager(model, stabilize_steps=500)
        for step in range(max_steps):
            warmup_mgr.step(step)
            loss = train_step(model, batch)
    """

    def __init__(
        self,
        model,
        stabilize_steps: int = 0,
        premix_lr_scale: float = 1.0,
        postmix_lr_scale: float = 1.0,
    ):
        self.model = model
        self.stabilize_steps = stabilize_steps
        self.premix_lr_scale = premix_lr_scale
        self.postmix_lr_scale = postmix_lr_scale
        self._current_phase = -1
        self._vha_layers = find_vha_attention_layers(model)

        if not self._vha_layers:
            logger.warning("No VHASelfAttention layers found. WarmupManager is a no-op.")

    def step(self, global_step: int):
        """Call each training step to manage phase transitions."""
        if self.stabilize_steps > 0 and global_step < self.stabilize_steps:
            if self._current_phase != 0:
                self._enter_phase(0)
        else:
            if self._current_phase != 1:
                self._enter_phase(1)

    def _enter_phase(self, phase: int):
        self._current_phase = phase
        if phase == 0:
            logger.info("VHA Warmup: phase 0 (stabilize) — VHA params frozen")
            for layer in self._vha_layers:
                for p in layer.vha_param_groups()["premix"] + layer.vha_param_groups()["postmix"]:
                    p.stop_gradient = True
        elif phase == 1:
            logger.info("VHA Warmup: phase 1 (activate) — all params unfrozen")
            for layer in self._vha_layers:
                for p in layer.vha_param_groups()["premix"] + layer.vha_param_groups()["postmix"]:
                    p.stop_gradient = False

    def get_vha_param_groups(self, base_lr: float) -> list[dict]:
        """Return VHA parameter groups with per-group LR for optimizer."""
        premix_params = []
        postmix_params = []
        for layer in self._vha_layers:
            groups = layer.vha_param_groups()
            premix_params.extend(groups["premix"])
            postmix_params.extend(groups["postmix"])

        param_groups = []
        if premix_params:
            param_groups.append({
                "params": premix_params,
                "learning_rate": base_lr * self.premix_lr_scale,
            })
        if postmix_params:
            param_groups.append({
                "params": postmix_params,
                "learning_rate": base_lr * self.postmix_lr_scale,
            })
        return param_groups

    @property
    def current_phase(self) -> int:
        return self._current_phase

    @property
    def num_vha_layers(self) -> int:
        return len(self._vha_layers)
