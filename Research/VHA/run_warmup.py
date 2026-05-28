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
VHA Warmup Training Entry Point.

A clean, standalone pipeline for VHA warmup training that:
  1. Creates model from config.json (no weights in model_name_or_path)
  2. Loads pre-converted VHA checkpoint (safetensors) directly into model
  3. Starts training — no flex_checkpoint, no implicit framework loading

Usage:
    python -m paddle.distributed.launch ... run_warmup.py config/qwen3/qwen3_vha_1p7B_warmup.json
"""

import glob
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Set

import numpy as np
import paddle
import paddlefleet

from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    print_rank_0,
)
from paddleformers.trainer import (
    PdArgumentParser,
    StepFlexToken,
    TrainingArguments,
    get_last_checkpoint,
)
from paddleformers.trainer.trainer import Trainer
from paddleformers.transformers import AutoTokenizer
from paddleformers.transformers.configuration_utils import LlmMetaConfig, llmmetaclass
from paddleformers.utils.batch_sampler import DistributedBatchSampler
from paddleformers.utils.log import logger

os.environ["USE_CASUAL_MASK"] = "True"

from models.qwen_provider import create_provider
from utils.warmup import VHAWarmupManager

from paddleformers.trainer.utils.doc import add_start_docstrings


# =============================================================================
# Arguments
# =============================================================================

@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class WarmupTrainingArguments(TrainingArguments):
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate decayed to."},
    )
    decay_steps: float = field(
        default=None,
        metadata={"help": "The steps to control learning rate decay."},
    )
    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Enable unified checkpoint format."},
    )
    recompute: bool = field(
        default=False,
        metadata={"help": "Recompute forward pass to save memory."},
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DataArguments:
    input_dir: str = field(
        default=None, metadata={"help": "Path to training data directory."}
    )
    max_seq_length: int = field(
        default=4096, metadata={"help": "Maximum sequence length."}
    )
    data_impl: str = field(default="mmap", metadata={"help": "Data format."})
    skip_warmup: bool = field(default=True, metadata={"help": "Skip mmap warmup."})
    data_cache: str = field(default=None, metadata={"help": "Data cache path."})
    share_folder: bool = field(default=False, metadata={"help": "Use shared folder."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to model config directory (config.json only, no weights)."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer path if different from model."}
    )
    continue_training: bool = field(
        default=False,
        metadata={"help": "Continue from existing weights or train from scratch."},
    )


@dataclass
class WarmupArguments:
    """Warmup-specific arguments."""
    init_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to VHA checkpoint directory (safetensors/pdparams) for weight initialization."},
    )
    stabilize_steps: int = field(
        default=0,
        metadata={"help": "Phase 0: steps with attn-adapt (qkv+o_proj+VHA train, rest frozen)."},
    )
    postmix_warmup_steps: int = field(
        default=0,
        metadata={"help": "Phase 1: steps to train all with premix/postmix lr_scale. 0 = skip phase 1."},
    )
    premix_lr_scale: float = field(
        default=1.0,
        metadata={"help": "Learning rate multiplier for premix parameters."},
    )
    postmix_lr_scale: float = field(
        default=1.0,
        metadata={"help": "Learning rate multiplier for postmix parameters."},
    )
    # Distillation
    teacher_model_path: str = field(
        default=None,
        metadata={"help": "Path to teacher (GQA) model config dir. Enables logit distillation."},
    )
    teacher_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to teacher checkpoint (safetensors/pdparams)."},
    )
    distill_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for distillation loss (0=pure CE, 1=pure KL)."},
    )
    distill_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for distillation softmax."},
    )
    layer_distill_beta: float = field(
        default=0.0,
        metadata={"help": "Weight for per-layer attention MSE loss. 0 = disabled."},
    )
    distill_end_step: int = field(
        default=0,
        metadata={"help": "Step at which distillation begins to decay. 0 = never stop."},
    )
    distill_decay_steps: int = field(
        default=0,
        metadata={"help": "Steps over which to linearly decay distillation after distill_end_step. 0 = hard cutoff."},
    )
    delta_debug_steps: str = field(
        default="",
        metadata={"help": "Comma-separated global steps for logging representative parameter deltas."},
    )


# =============================================================================
# Checkpoint Loading
# =============================================================================

def model_provider_num_layers(model) -> int:
    for _, sublayer in model.named_sublayers():
        if hasattr(sublayer, "config") and hasattr(sublayer.config, "num_hidden_layers"):
            return int(sublayer.config.num_hidden_layers)
    if hasattr(model, "_layers"):
        return max(len(model._layers) - 3, 0)
    return 28


def load_checkpoint_into_model(model, checkpoint_dir: str):
    """
    Load VHA checkpoint (safetensors or pdparams) directly into model state_dict.

    Key mapping strategy:
      - Try exact match first
      - If no exact match, try stripping/adding common suffixes (.weight, .w_0)
      - For VHA params (vha_premix_weight, vha_postmix_U/V), handle both
        'param_name' and 'param_name.weight' formats

    Args:
        model: The VHA model (from vha_gpt_builder)
        checkpoint_dir: Directory containing *.safetensors or *.pdparams files
    """
    # Discover checkpoint files
    pdparams_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pdparams")))
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

    if pdparams_files:
        logger.info(f"[Warmup] Loading from pdparams: {pdparams_files}")
        state_dict = {}
        for path in pdparams_files:
            state_dict.update(paddle.load(path))
    elif safetensors_files:
        logger.info(f"[Warmup] Loading from safetensors: {safetensors_files}")
        from safetensors import safe_open
        import torch
        state_dict = {}
        for sf_path in safetensors_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    # Convert torch bf16 -> float32 numpy (paddle can cast later)
                    state_dict[key] = t.float().numpy()
        del torch
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            "Expected *.pdparams or *.safetensors files."
        )

    logger.info(f"[Warmup] Checkpoint has {len(state_dict)} parameters")

    # GPTModel.state_dict() exposes semantic checkpoint names, while
    # named_parameters() exposes internal PipelineLayer names. Use set_state_dict()
    # so PaddleFleet preserves its own pipeline/shared-weight mapping.
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    logger.info(f"[Warmup] Model has {len(model_keys)} parameters")

    ckpt_sample = list(state_dict.keys())[:5]
    model_sample = list(model_keys)[:5]
    logger.info(f"[Warmup] Checkpoint key samples: {ckpt_sample}")
    logger.info(f"[Warmup] Model key samples: {model_sample}")

    loaded, skipped, shape_mismatch = 0, 0, 0
    matched_model_keys = set()
    new_model_state = {}

    for ckpt_key, ckpt_value in state_dict.items():
        # Generate candidate model keys
        candidates = [ckpt_key]

        # Try adding/stripping "model." prefix
        if not ckpt_key.startswith("model."):
            candidates.append("model." + ckpt_key)
        else:
            candidates.append(ckpt_key[len("model."):])

        # PipelineLayer names layers by stage index: embedding=0, transformer layers=1..N,
        # final norm=N+1, lm_head=N+2. Converted checkpoints use semantic names.
        semantic_key = ckpt_key[len("model."):] if ckpt_key.startswith("model.") else ckpt_key
        if semantic_key.startswith("embedding."):
            candidates.append(f"0.{semantic_key}")
        elif semantic_key.startswith("layers."):
            parts = semantic_key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                candidates.append(f"{int(parts[1]) + 1}.{parts[2]}")
        elif semantic_key == "norm.weight":
            candidates.append(f"{model_provider_num_layers(model) + 1}.norm.weight")
        elif semantic_key == "lm_head.weight":
            candidates.append(f"{model_provider_num_layers(model) + 2}.weight")
            candidates.append("model.lm_head.weight")
            candidates.append("model.shared_head.weight")

        # Try adding .w_0 suffix (paddlefleet create_parameter naming)
        if not ckpt_key.endswith(".w_0"):
            candidates.append(ckpt_key + ".w_0")
            if not ckpt_key.startswith("model."):
                candidates.append("model." + ckpt_key + ".w_0")

        # Try stripping .w_0 suffix
        if ckpt_key.endswith(".w_0"):
            candidates.append(ckpt_key[:-4])

        # Try adding .weight suffix
        if not ckpt_key.endswith(".weight"):
            candidates.append(ckpt_key + ".weight")
            if not ckpt_key.startswith("model."):
                candidates.append("model." + ckpt_key + ".weight")

        # Try stripping .weight suffix
        if ckpt_key.endswith(".weight"):
            candidates.append(ckpt_key[:-7])

        matched = False
        for candidate in candidates:
            if candidate in model_keys and candidate not in matched_model_keys:
                # Shape check
                model_shape = list(model_state[candidate].shape)
                ckpt_shape = list(ckpt_value.shape)
                if model_shape == ckpt_shape:
                    new_model_state[candidate] = paddle.to_tensor(ckpt_value).cast(model_state[candidate].dtype)
                    matched_model_keys.add(candidate)
                    loaded += 1
                    matched = True
                    break
                else:
                    shape_mismatch += 1
                    logger.debug(
                        f"[Warmup] Shape mismatch for {ckpt_key} -> {candidate}: "
                        f"ckpt={ckpt_shape} vs model={model_shape}"
                    )

        if not matched:
            skipped += 1
            if skipped <= 10:
                logger.warning(f"[Warmup] Skipped (no match): {ckpt_key}")

    model.set_state_dict(new_model_state)

    del state_dict
    del model_state
    del new_model_state
    import gc
    gc.collect()
    paddle.device.cuda.empty_cache()

    unloaded_model_keys = model_keys - matched_model_keys
    logger.info(
        f"[Warmup] Checkpoint loading complete: "
        f"{loaded} loaded, {skipped} skipped, {shape_mismatch} shape mismatches"
    )
    if unloaded_model_keys:
        logger.info(
            f"[Warmup] {len(unloaded_model_keys)} model params not in checkpoint "
            f"(using random init). Samples: {list(unloaded_model_keys)[:10]}"
        )

    return loaded


# =============================================================================
# Data
# =============================================================================

def create_pretrained_dataset(data_args, training_args, data_file, tokenizer, need_data=True):
    train_val_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        0,
        0,
    ]

    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))

    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl=data_args.data_impl,
        splits_string="1,0,0",
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_length,
        seed=training_args.seed,
        skip_warmup=data_args.skip_warmup,
        share_folder=data_args.share_folder,
        data_cache_path=data_args.data_cache,
        need_data=need_data,
    )

    from paddleformers.data import Stack

    def _collate_data(batch, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in batch])
        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]
        return {"input_ids": tokens, "labels": labels}

    return train_dataset, valid_dataset, test_dataset, _collate_data


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        return args.input_dir.split()
    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f)) and ("_idx.npz" in str(f) or ".idx" in str(f)))
    ]
    files = [x.replace("_idx.npz", "") for x in files]
    files = [x.replace(".idx", "") for x in files]
    if len(files) > 1:
        ret = []
        for x in files:
            ret.append(1.0)
            ret.append(x)
        return ret
    return files


def probe_loaded_model_ce(model, train_dataset, data_collator, batch_size: int = 1):
    was_training = model.training
    model.eval()
    samples = [train_dataset[i] for i in range(batch_size)]
    batch = data_collator(samples)
    input_ids = paddle.to_tensor(batch["input_ids"])
    labels = paddle.to_tensor(batch["labels"])
    with paddle.no_grad():
        logits = model({"input_ids": input_ids})
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = model._loss_fn[0](logits, labels)
    logger.info(
        f"[WARMUP_PROBE_CE] input_shape={list(input_ids.shape)} "
        f"label_shape={list(labels.shape)} loss={float(loss.numpy()):.6f} "
        f"first_input={input_ids.numpy().reshape([-1])[:8].tolist()} "
        f"first_label={labels.numpy().reshape([-1])[:8].tolist()}"
    )
    if was_training:
        model.train()
    return float(loss.numpy())


# =============================================================================
# Trainer
# =============================================================================

class WarmupTrainer(Trainer):
    """Trainer with VHA warm-up phase support and optional logit distillation."""

    def __init__(self, *args, warmup_manager: Optional[VHAWarmupManager] = None,
                 teacher_model=None, layer_distill=None, distill_end_step: int = 0, distill_decay_steps: int = 0,
                 delta_debug_steps: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.is_pretraining = True
        self.warmup_manager = warmup_manager
        self.teacher_model = teacher_model
        self.layer_distill = layer_distill
        self.distill_end_step = distill_end_step
        self.distill_decay_steps = distill_decay_steps
        self.delta_debug_steps: Set[int] = {
            int(step.strip()) for step in delta_debug_steps.split(",") if step.strip()
        }
        self._delta_debug_refs = {}

    def _select_delta_debug_params(self, model):
        selected = {}
        patterns = {
            "embed": "embedding.embed_tokens.weight",
            "mlp": "mlp.up_gate_proj.weight",
            "norm": "input_layernorm.weight",
            "attn": "self_attn.qkv_proj.weight",
            "vha": "self_attn.vha_postmix_U",
        }
        for name, param in model.named_parameters():
            for group, pattern in patterns.items():
                if group not in selected and pattern in name:
                    selected[group] = (name, param)
        return selected

    def _log_delta_debug(self, model):
        step = int(self.state.global_step)
        if step not in self.delta_debug_steps:
            return
        selected = self._select_delta_debug_params(model)
        if not self._delta_debug_refs:
            for group, (_, param) in selected.items():
                self._delta_debug_refs[group] = param.detach().cast("float32").clone()
            logger.info(f"[DELTA_DEBUG] step={step} initialized refs groups={sorted(self._delta_debug_refs)}")
            return
        parts = []
        for group, (name, param) in selected.items():
            ref = self._delta_debug_refs.get(group)
            if ref is None:
                continue
            current = param.detach().cast("float32")
            delta = paddle.linalg.norm(current - ref).item()
            param_norm = paddle.linalg.norm(current).item()
            scale = getattr(param, "optimize_attr", {}).get("learning_rate", 1.0)
            stop_gradient = getattr(param, "stop_gradient", None)
            parts.append(
                f"{group}:{delta:.8e}/norm={param_norm:.8e}/scale={scale}/stop_gradient={stop_gradient}/name={name}"
            )
        phase = self.warmup_manager.current_phase if self.warmup_manager is not None else -1
        logger.info(f"[DELTA_DEBUG] step={step} phase={phase} " + " | ".join(parts))

    def training_step(self, model, inputs):
        if self.warmup_manager is not None:
            self.warmup_manager.step(self.state.global_step)
        self._log_delta_debug(model)

        # Compute distillation scale (1.0 = full, 0.0 = off)
        distill_scale = self._get_distill_scale()

        # Enable capture for layer-wise distillation
        if distill_scale > 0 and self.layer_distill is not None:
            self.layer_distill.begin_capture()

        # Run teacher forward and store logits for distillation loss
        if distill_scale > 0:
            self._run_teacher_forward(inputs)

        # Update DistillationLoss scale factor
        if self.teacher_model is not None:
            from utils.distillation import DistillationLoss
            DistillationLoss.set_scale(distill_scale)

        return super().training_step(model, inputs)

    def _get_distill_scale(self) -> float:
        """Compute current distillation scale: 1.0 (full) -> 0.0 (off)."""
        if self.teacher_model is None:
            return 0.0
        if self.distill_end_step <= 0:
            return 1.0  # Never decay

        step = self.state.global_step
        if step < self.distill_end_step:
            return 1.0  # Before decay starts

        if self.distill_decay_steps <= 0:
            return 0.0  # Hard cutoff (legacy behavior)

        # Linear decay over distill_decay_steps
        decay_progress = (step - self.distill_end_step) / self.distill_decay_steps
        if decay_progress >= 1.0:
            return 0.0
        return 1.0 - decay_progress

    def _run_teacher_forward(self, inputs):
        """Run teacher forward pass, store logits in DistillationLoss buffer."""
        from utils.distillation import DistillationLoss

        with paddle.no_grad():
            # Teacher's PipelineLayer.forward() expects a dict with "input_ids"
            # (same format as GPTEmbedding.forward receives)
            if isinstance(inputs, dict):
                teacher_input = {"input_ids": inputs["input_ids"]}
            elif isinstance(inputs, (list, tuple)):
                first = inputs[0]
                if isinstance(first, dict):
                    teacher_input = {"input_ids": first["input_ids"]}
                else:
                    teacher_input = {"input_ids": first}
            else:
                teacher_input = {"input_ids": inputs}

            # PipelineLayer.forward() runs all layers sequentially for pp_size=1
            # Returns logits tensor [B, S, V] from GPTLMHead
            teacher_logits = self.teacher_model(teacher_input)
            if isinstance(teacher_logits, tuple):
                teacher_logits = teacher_logits[0]

            DistillationLoss.set_teacher_logits(teacher_logits.detach())

    def create_optimizer(self, model=None):
        """Override to apply per-group LR for VHA params."""
        opt = super().create_optimizer(model)

        # Apply lr_scale to VHA parameter groups if configured
        if self.warmup_manager is not None:
            premix_scale = self.warmup_manager.premix_lr_scale
            postmix_scale = self.warmup_manager.postmix_lr_scale
            if premix_scale != 1.0 or postmix_scale != 1.0:
                # Collect VHA param ids for fast lookup
                premix_ids = set()
                postmix_ids = set()
                for layer in self.warmup_manager._vha_layers:
                    groups = layer.vha_param_groups()
                    for p in groups["premix"]:
                        premix_ids.add(id(p))
                    for p in groups["postmix"]:
                        postmix_ids.add(id(p))

                # Set per-param lr multiplier via optimize_attr
                for p in self.model.parameters():
                    if id(p) in premix_ids:
                        p.optimize_attr = {"learning_rate": premix_scale}
                    elif id(p) in postmix_ids:
                        p.optimize_attr = {"learning_rate": postmix_scale}

                logger.info(
                    f"[Warmup] Applied lr_scale: premix={premix_scale}, postmix={postmix_scale}"
                )
        return opt

    def _get_train_sampler(self):
        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )


# =============================================================================
# Utilities
# =============================================================================

def _set_random_seed(seed_: int):
    if seed_ is not None and seed_ > 0:
        seed = seed_ + (100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
            paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, WarmupTrainingArguments, WarmupArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, warmup_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args, warmup_args = parser.parse_args_into_dataclasses()

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    paddle.set_device(training_args.device)
    _set_random_seed(seed_=training_args.seed)

    training_args.eval_iters = 10
    training_args.test_iters = training_args.eval_iters * 10

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(warmup_args, "Warmup")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"bf16: {training_args.bf16}"
    )

    # Detect last checkpoint (for resume)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    # Model — create from config.json only (no weights loaded by framework)
    model_provider = create_provider(model_args.model_name_or_path)
    model_provider.seq_length = data_args.max_seq_length
    model_provider.max_sequence_length = data_args.max_seq_length

    logger.info(f"Creating model from config: {model_args.model_name_or_path}")
    model = model_provider.provide()

    if training_args.recompute:
        def _enable_recompute(layer):
            if hasattr(layer, "enable_recompute") and not layer.enable_recompute:
                layer.enable_recompute = True
        model.apply(_enable_recompute)

    # Load VHA checkpoint weights directly (the key step that avoids OOM)
    if warmup_args.init_checkpoint is not None:
        logger.info(f"=== Loading VHA init checkpoint: {warmup_args.init_checkpoint} ===")
        n_loaded = load_checkpoint_into_model(model, warmup_args.init_checkpoint)
        if n_loaded == 0:
            raise RuntimeError(
                f"No parameters loaded from {warmup_args.init_checkpoint}. "
                "Check key format mismatch between checkpoint and model."
            )
        logger.info(f"=== VHA init checkpoint loaded successfully ({n_loaded} params) ===")
    else:
        logger.warning("No init_checkpoint specified — training from random initialization!")

    # VHA warmup manager (freeze/unfreeze scheduling)
    warmup_manager = VHAWarmupManager(
        model,
        stabilize_steps=warmup_args.stabilize_steps,
        postmix_warmup_steps=warmup_args.postmix_warmup_steps,
        premix_lr_scale=warmup_args.premix_lr_scale,
        postmix_lr_scale=warmup_args.postmix_lr_scale,
    )
    logger.info(
        f"VHA WarmupManager: {warmup_manager.num_vha_layers} layers, "
        f"stabilize_steps={warmup_args.stabilize_steps}, "
        f"postmix_warmup_steps={warmup_args.postmix_warmup_steps}"
    )

    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    # --- Teacher model for distillation (optional) ---
    teacher_model = None
    if warmup_args.teacher_model_path is not None and warmup_args.teacher_model_path != "None" and warmup_args.teacher_model_path:
        from utils.distillation import DistillationLoss

        logger.info(f"=== Loading teacher model for distillation ===")
        logger.info(f"  Teacher config: {warmup_args.teacher_model_path}")
        logger.info(f"  alpha={warmup_args.distill_alpha}, T={warmup_args.distill_temperature}")

        DistillationLoss.configure(
            alpha=warmup_args.distill_alpha,
            temperature=warmup_args.distill_temperature,
        )

        # Build teacher model (GQA)
        teacher_provider = create_provider(warmup_args.teacher_model_path)
        teacher_provider.seq_length = data_args.max_seq_length
        teacher_provider.max_sequence_length = data_args.max_seq_length
        teacher_model = teacher_provider.provide()

        # Load teacher checkpoint
        if warmup_args.teacher_checkpoint is not None:
            n_teacher = load_checkpoint_into_model(teacher_model, warmup_args.teacher_checkpoint)
            logger.info(f"  Teacher checkpoint loaded ({n_teacher} params)")
        else:
            logger.warning("  No teacher_checkpoint — teacher uses random weights!")

        # Freeze teacher
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.stop_gradient = True

        # Replace student's loss_fn with DistillationLoss
        if hasattr(model, '_loss_fn') and model._loss_fn:
            distill_loss = DistillationLoss(model._loss_fn[0].config)
            model._loss_fn = [distill_loss]
            logger.info("  Student loss_fn -> DistillationLoss")

        logger.info("=== Teacher model ready ===")

    # --- Layer-wise attention distillation (optional) ---
    layer_distill = None
    if teacher_model is not None and warmup_args.layer_distill_beta > 0:
        from utils.distillation import LayerAttnDistillation

        layer_distill = LayerAttnDistillation(
            student_model=model,
            teacher_model=teacher_model,
            beta=warmup_args.layer_distill_beta,
        )
        logger.info(
            f"Layer-wise attention distillation: beta={warmup_args.layer_distill_beta}, "
            f"layers={layer_distill.num_layers}"
        )

    # Data
    data_file = get_train_data_file(data_args)
    train_dataset, _, _, data_collator = create_pretrained_dataset(
        data_args, training_args, data_file, tokenizer,
        need_data=training_args.should_load_dataset,
    )

    if (
        os.environ.get("WARMUP_PROBE_CE", "0") == "1"
        and paddle.distributed.get_rank() == 0
        and training_args.should_load_dataset
    ):
        probe_loaded_model_ce(model, train_dataset, data_collator, batch_size=1)

    callbacks = [StepFlexToken()]

    # Trainer
    trainer = WarmupTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        optimizers=(None, None),
        tokenizer=tokenizer,
        callbacks=callbacks,
        warmup_manager=warmup_manager,
        teacher_model=teacher_model,
        layer_distill=layer_distill,
        distill_end_step=warmup_args.distill_end_step,
        distill_decay_steps=warmup_args.distill_decay_steps,
        delta_debug_steps=warmup_args.delta_debug_steps,
    )

    # Resume from training checkpoint (not init checkpoint)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Train
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_train and training_args.should_load_dataset:
        total_effective_tokens = (
            training_args.per_device_train_batch_size
            * training_args.dataset_world_size
            * training_args.max_steps
            * training_args.gradient_accumulation_steps
            * data_args.max_seq_length
        )
        effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
        logger.info(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
