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
UHA pre-training entry point.

Reuses PaddleFleet/PaddleFormers training components for:
  - GQA baseline training
  - UHA training with unequal QK/V heads
  - Checkpoint save/resume
  - Distributed training (TP/PP/DP via PaddleFleet)
"""

import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

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

from paddleformers.trainer.utils.doc import add_start_docstrings


# =============================================================================
# Arguments
# =============================================================================

@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
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
        metadata={"help": "Path to pretrained model or tokenizer."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer path if different from model."}
    )
    continue_training: bool = field(
        default=False,
        metadata={"help": "Continue from existing weights or train from scratch."},
    )


@dataclass
class UHAArguments:
    """UHA-specific training arguments."""
    uha_use_triton_kernel: bool = field(
        default=True,
        metadata={"help": "Use UHA Triton causal attention kernel when supported."},
    )


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


# =============================================================================
# Trainer
# =============================================================================

class PretrainingTrainer(Trainer):
    """PaddleFleet/PaddleFormers trainer for UHA pre-training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_pretraining = True

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
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments, UHAArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, uha_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args, uha_args = parser.parse_args_into_dataclasses()

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
    training_args.print_config(uha_args, "UHA")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"bf16: {training_args.bf16}"
    )

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    # Model — load architecture from config.json
    model_provider = create_provider(model_args.model_name_or_path)
    model_provider.seq_length = data_args.max_seq_length
    model_provider.max_sequence_length = data_args.max_seq_length
    model_provider.uha_use_triton_kernel = uha_args.uha_use_triton_kernel

    logger.info(f"Creating model from: {model_args.model_name_or_path}")
    model = model_provider.provide()

    if training_args.recompute:
        def _enable_recompute(layer):
            if hasattr(layer, "enable_recompute") and not layer.enable_recompute:
                layer.enable_recompute = True
        model.apply(_enable_recompute)

    logger.info(
        f"UHA config: qk_heads={model_provider.num_attention_heads}, "
        f"value_heads={model_provider.uha_num_value_heads}, "
        f"value_head_dim={model_provider.uha_value_head_dim}, "
        f"use_triton_kernel={model_provider.uha_use_triton_kernel}"
    )

    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    # Data
    data_file = get_train_data_file(data_args)
    train_dataset, _, _, data_collator = create_pretrained_dataset(
        data_args, training_args, data_file, tokenizer,
        need_data=training_args.should_load_dataset,
    )

    total_effective_tokens = (
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps
        * data_args.max_seq_length
    )

    callbacks = [StepFlexToken()]

    # Trainer
    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        optimizers=(None, None),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Resume
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
        effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
        logger.info(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
