# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""DHA-style fusion training entry point (Trainer-based).

Pipeline:
  1. Build GQA pipeline model via existing provider, with a custom
     DHALanguageLoss as loss_fn so the constraint enters the same
     forward graph as the LM loss (preserves recompute compatibility
     and pp pipeline schedule).
  2. Load GQA pretrain ckpt.
  3. Install DHA fusion hooks (omega gates + postmix UV) per attention
     layer; populate the loss_fn\'s fusion_states reference.
  4. Run Trainer.train(). LM loss + lambda * max(C-t, 0) is built inside
     the loss layer; backward proceeds normally through the pipeline.
  5. ALMCallback updates lambda + records history (only syncs C->CPU on
     log/dual-update steps).

The checkpoint contains base weights + fusion params (omega_*_logits,
postmix_U/V). Use fold.py to collapse into a 2-head VHA at inference time.
"""

import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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
    TrainerCallback,
    TrainingArguments,
    get_last_checkpoint,
)
from paddleformers.trainer.trainer import Trainer
from paddleformers.transformers import AutoTokenizer
from paddleformers.transformers.configuration_utils import llmmetaclass
from paddleformers.utils.batch_sampler import DistributedBatchSampler
from paddleformers.utils.log import logger
from paddleformers.trainer.utils.doc import add_start_docstrings

os.environ["USE_CASUAL_MASK"] = "True"

# ---- local imports ----
_DHA_DIR = os.path.dirname(os.path.abspath(__file__))
if _DHA_DIR not in sys.path:
    sys.path.insert(0, _DHA_DIR)
_VHA_DIR = os.path.normpath(os.path.join(_DHA_DIR, "..", "..", "..", "VHA"))
if _VHA_DIR not in sys.path:
    sys.path.insert(0, _VHA_DIR)

from models.qwen_provider import create_provider                # noqa: E402
from paddlefleet.transformer.transformer_layer import TransformerLayer  # noqa: E402
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss  # noqa: E402

from grouping import load_groupings, validate_groupings         # noqa: E402
from attention_patch import install_all_fusion_hooks            # noqa: E402
from alm_loss import (                                          # noqa: E402
    ALMConfig, ALMState, compute_total_constraint,
)


# =============================================================================
# Arguments
# =============================================================================

@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    min_learning_rate: float = field(default=1e-5)
    decay_steps: float = field(default=None)
    unified_checkpoint: bool = field(default=True)
    recompute: bool = field(default=False)


@dataclass
class DataArguments:
    input_dir: str = field(default=None)
    max_seq_length: int = field(default=4096)
    data_impl: str = field(default="mmap")
    skip_warmup: bool = field(default=True)
    data_cache: str = field(default=None)
    share_folder: bool = field(default=False)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    continue_training: bool = field(default=False)


@dataclass
class DHAArguments:
    gqa_checkpoint: str = field(default=None)
    groupings_path: str = field(default=None)
    fusion_decay_steps: int = field(default=1500)
    lambda_init: float = field(default=0.0)
    lambda_lr: float = field(default=1.0)
    lambda_max: float = field(default=100.0)
    dual_update_interval: int = field(default=50)
    omega_init_path: Optional[str] = field(default=None)
    omega_init_temperature: float = field(default=5.0)


# =============================================================================
# DHA Language Loss: LM loss + ALM constraint, in-graph.
# =============================================================================

class DHALanguageLoss(LanguageLoss):
    """LanguageLoss + ALM-augmented DHA constraint.

    Adds  lambda * max(C - t, 0)  to the LM loss, where C is the per-layer
    intra-group K/V relative MSE computed from the FusionStates that have
    just been populated by forward_pre_hooks during this same forward pass.

    The constraint tensor is created inside the loss layer\'s forward, so it
    is part of the pipeline\'s loss graph and survives normal backward.
    """

    # class-level scratch for callbacks (last-step scalars, detached)
    last_lm_loss: paddle.Tensor | None = None
    last_constraint: paddle.Tensor | None = None
    last_total: paddle.Tensor | None = None

    def __init__(self, config, fusion_states_holder: list, alm_state: ALMState, pg_collection=None):
        super().__init__(config=config, pg_collection=pg_collection)
        # holder is mutated AFTER hooks are installed
        self.fusion_states_holder = fusion_states_holder
        self.alm = alm_state

    def forward(self, logits, labels):
        lm_loss = super().forward(logits, labels)

        # Compute DHA constraint from cached pre-fusion K/V (same forward pass).
        if self.fusion_states_holder:
            constraint = compute_total_constraint(self.fusion_states_holder)
            t = self.alm.current_target()
            violation = paddle.clip(constraint.cast(lm_loss.dtype) - t, min=0.0)
            total = lm_loss + self.alm.lam * violation
        else:
            constraint = paddle.zeros([], dtype=lm_loss.dtype)
            total = lm_loss

        # Stash detached scalars for callback (no GPU sync; just tensor refs).
        DHALanguageLoss.last_lm_loss = lm_loss.detach()
        DHALanguageLoss.last_constraint = constraint.detach()
        DHALanguageLoss.last_total = total.detach()
        return total


# =============================================================================
# GQA checkpoint loader (safetensors -> pipeline-renamed params)
# =============================================================================

def _ckpt_key_to_pipeline_name(key):
    k = key.replace("model.", "", 1) if key.startswith("model.") else key
    if k.startswith("embedding."):
        return f"0.{k}"
    if k.startswith("layers."):
        parts = k.split(".", 2)
        layer_idx = int(parts[1])
        return f"{layer_idx + 1}.{parts[2]}"
    if k == "norm.weight":
        return "29.norm.weight"
    if k == "lm_head.weight":
        return "30.weight"
    return None


def load_gqa_safetensors(model, ckpt_dir):
    from safetensors import safe_open
    param_dict = dict(model.named_parameters())
    sf_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
    loaded = 0
    shape_mismatch = []
    for sf in sf_files:
        with safe_open(os.path.join(ckpt_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                val = f.get_tensor(key).float().numpy()
                pname = _ckpt_key_to_pipeline_name(key)
                param = param_dict.get(pname) if pname else None
                if param is None:
                    for cand in [key, "model." + key, key.replace("model.", "")]:
                        if cand in param_dict:
                            param = param_dict[cand]
                            break
                if param is None:
                    continue
                if list(param.shape) != list(val.shape):
                    shape_mismatch.append((key, list(val.shape), list(param.shape)))
                    continue
                param.set_value(paddle.to_tensor(val).cast(param.dtype))
                loaded += 1
    if shape_mismatch:
        for k, vs, ps in shape_mismatch[:5]:
            logger.warning(f"  shape mismatch: {k} ckpt={vs} model={ps}")
        raise RuntimeError(f"{len(shape_mismatch)} shape-mismatched params; refusing to continue")
    return loaded, len(param_dict)


# =============================================================================
# Trainer + ALM callback
# =============================================================================

class DHAFusionTrainer(Trainer):
    """Standard Trainer; just pins the dataset sampler to match run_pretrain.py."""

    def _get_train_sampler(self):
        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )


class ALMCallback(TrainerCallback):
    """Updates lambda and records ALM history.

    Uses tensor refs cached on DHALanguageLoss to avoid extra .item() syncs;
    only forces sync to CPU on dual-update / log steps.
    """

    def __init__(self, alm_state: ALMState, dual_interval: int, log_interval: int):
        self.alm = alm_state
        self.dual_interval = max(1, dual_interval)
        self.log_interval = max(1, log_interval)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        is_dual = (step > 0 and step % self.dual_interval == 0)
        is_log = (step > 0 and step % self.log_interval == 0)
        if not (is_dual or is_log):
            self.alm.step = step
            return

        if DHALanguageLoss.last_constraint is None:
            return
        c_val = float(DHALanguageLoss.last_constraint.cast("float32").item())
        lm_val = float(DHALanguageLoss.last_lm_loss.cast("float32").item())

        self.alm.step = step
        if is_dual:
            self.alm.maybe_update_lambda(c_val)
        self.alm.end_step(c_val, lm_val)
        if is_log and args.local_rank in (0, -1):
            logger.info(
                f"[ALM] step={step} lm={lm_val:.4f} C={c_val:.5f} "
                f"t={self.alm.current_target():.5f} lam={self.alm.lam:.3f}"
            )


# =============================================================================
# Data
# =============================================================================

def create_pretrained_dataset(data_args, training_args, data_file, tokenizer, need_data=True):
    n_samples = (
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps
    )
    print_rank_0(f" > target train samples: {n_samples}")

    train_ds, _, _ = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl=data_args.data_impl,
        splits_string="1,0,0",
        train_val_test_num_samples=[n_samples, 0, 0],
        seq_length=data_args.max_seq_length,
        seed=training_args.seed,
        skip_warmup=data_args.skip_warmup,
        share_folder=data_args.share_folder,
        data_cache_path=data_args.data_cache,
        need_data=need_data,
    )
    from paddleformers.data import Stack
    def _collate(batch, stack_fn=Stack()):
        toks = stack_fn([x["text"] for x in batch])
        return {"input_ids": toks[:, :-1], "labels": toks[:, 1:]}
    return train_ds, _collate


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        return args.input_dir.split()
    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f))
            and ("_idx.npz" in str(f) or ".idx" in str(f)))
    ]
    files = [x.replace("_idx.npz", "").replace(".idx", "") for x in files]
    if len(files) > 1:
        ret = []
        for x in files:
            ret.append(1.0); ret.append(x)
        return ret
    return files


# =============================================================================
# Main
# =============================================================================


def build_cosine_omega_init(
    cosine_path: str,
    diagnostics_groupings,
    temperature: float,
):
    """Build per-layer (omega_k_logits, omega_v_logits) ndarrays from cosine
    similarity matrices. logits[h, g] = T * mean_{m in members(g)} cos[h, m]."""
    import json as _json
    with open(cosine_path) as f:
        cos_data = _json.load(f)
    n_layers = len(diagnostics_groupings)
    if "layers" not in cos_data or len(cos_data["layers"]) != n_layers:
        raise ValueError(
            f"cosine file has {len(cos_data.get('layers', []))} layers, "
            f"groupings has {n_layers}"
        )
    out = []
    for li in range(n_layers):
        cos_k = np.asarray(cos_data["layers"][li]["cos_k"], dtype=np.float32)
        cos_v = np.asarray(cos_data["layers"][li]["cos_v"], dtype=np.float32)
        groups = diagnostics_groupings[li]
        n_src = sum(len(g) for g in groups)
        n_groups = len(groups)
        if cos_k.shape != (n_src, n_src) or cos_v.shape != (n_src, n_src):
            raise ValueError(
                f"layer {li}: cos shape {cos_k.shape}/{cos_v.shape} != ({n_src},{n_src})"
            )
        def _logits(cos_mat):
            logits = np.zeros((n_src, n_groups), dtype=np.float32)
            for gi, members in enumerate(groups):
                centroid_cos = cos_mat[:, members].mean(axis=1)
                logits[:, gi] = temperature * centroid_cos
            return logits
        out.append((_logits(cos_k), _logits(cos_v)))
    return out


def _set_random_seed(seed_):
    if seed_ is None or seed_ <= 0:
        raise ValueError("seed must be positive integer")
    seed = seed_ + 100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
        paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PreTrainingArguments, DHAArguments)
    )
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, dha_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args, dha_args = parser.parse_args_into_dataclasses()

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
    training_args.print_config(dha_args, "DHA")

    last_ckpt = None
    if (os.path.isdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming from {last_ckpt}")

    # Recompute is incompatible with our forward-hook caching strategy:
    # the cached K/V are populated on the first forward pass; with recompute
    # the autograd graph attached to those tensors is dropped after the layer
    # returns, so constraint backward through cached refs would fail.
    if training_args.recompute:
        logger.warning(
            "recompute=True is incompatible with DHA fusion hooks; forcing False."
        )
        training_args.recompute = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    # ---- Provider + custom loss_fn (constraint injected in-graph) ----
    provider = create_provider(model_args.model_name_or_path)
    provider.seq_length = data_args.max_seq_length
    provider.max_sequence_length = data_args.max_seq_length

    alm_cfg = ALMConfig(
        target_initial=-1.0,                   # auto-tune from first observed C
        target_decay_steps=dha_args.fusion_decay_steps,
        lambda_init=dha_args.lambda_init,
        lambda_lr=dha_args.lambda_lr,
        lambda_max=dha_args.lambda_max,
        dual_update_interval=dha_args.dual_update_interval,
    )
    alm_state = ALMState(alm_cfg)
    fusion_states_holder: List = []  # populated AFTER hook install

    dha_loss_fn = DHALanguageLoss(
        config=provider,
        fusion_states_holder=fusion_states_holder,
        alm_state=alm_state,
    )

    logger.info(f"Building model: {model_args.model_name_or_path}")
    model = provider.provide(loss_fn=dha_loss_fn)

    # ---- Load GQA weights (skip if resuming) ----
    if last_ckpt is None and training_args.resume_from_checkpoint is None:
        if dha_args.gqa_checkpoint is None:
            raise ValueError("dha_args.gqa_checkpoint required when not resuming.")
        n, t = load_gqa_safetensors(model, dha_args.gqa_checkpoint)
        logger.info(f"Loaded {n}/{t} GQA params from {dha_args.gqa_checkpoint}")

    # ---- Install fusion hooks ----
    layers = [fn for fn in model.run_function if isinstance(fn, TransformerLayer)]
    num_layers = len(layers)
    logger.info(f"#transformer layers: {num_layers}")
    groupings = load_groupings(dha_args.groupings_path)
    validate_groupings(groupings, n_layers=num_layers, n_src_heads=8, n_groups=2)
    if dha_args.omega_init_path:
        omega_init_per_layer = build_cosine_omega_init(
            dha_args.omega_init_path,
            groupings,
            temperature=dha_args.omega_init_temperature,
        )
        logger.info(
            f"omega init: cosine-weighted from {dha_args.omega_init_path} "
            f"(T={dha_args.omega_init_temperature})"
        )
    else:
        omega_init_per_layer = None
        logger.warning(
            "omega init: zero (uniform-after-softmax mean-pooling start). "
            "Set dha_args.omega_init_path for cosine init."
        )
    fusion_states = install_all_fusion_hooks(
        layers, groupings,
        attention_attr="self_attn",
        core_attention_attr="core_attention",
        kv_arg_indices=(1, 2),
        n_groups=2, total_q_heads=16, head_dim=128, postmix_rank=4,
        omega_init_per_layer=omega_init_per_layer,
    )
    fusion_states_holder.extend(fusion_states)
    logger.info(f"Installed fusion hooks on {len(fusion_states)} layers")

    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    # ---- Data ----
    data_files = get_train_data_file(data_args)
    train_ds, collator = create_pretrained_dataset(
        data_args, training_args, data_files, tokenizer,
        need_data=training_args.should_load_dataset,
    )

    # ---- Trainer ----
    alm_callback = ALMCallback(
        alm_state=alm_state,
        dual_interval=dha_args.dual_update_interval,
        log_interval=max(1, training_args.logging_steps),
    )
    trainer = DHAFusionTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds if training_args.do_train else None,
        optimizers=(None, None),
        tokenizer=tokenizer,
        callbacks=[StepFlexToken(), alm_callback],
    )

    ckpt = training_args.resume_from_checkpoint or last_ckpt
    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=ckpt)
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        if training_args.local_rank in (0, -1):
            import json
            with open(os.path.join(training_args.output_dir, "alm_history.json"), "w") as f:
                json.dump(alm_state.history, f, indent=2)


if __name__ == "__main__":
    main()
