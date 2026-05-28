#!/usr/bin/env python3
"""DHA-style fusion training for VHA: GQA(8KV) -> [fusion] -> VHA(2KV+postmix).

Loads a GQA checkpoint, attaches per-layer fusion gates (omega_K, omega_V) and
postmix UV residuals via forward hooks on each layer's core_attention, and
trains all parameters jointly with:

    L = L_lm + lambda * max(L_fusion - t(s), 0)

where L_fusion is the per-layer intra-group MSE on K, V activations and
lambda is updated via ALM dual ascent. Target t(s) decays linearly to 0 over
`fusion_decay_steps`, after which the constraint is hard.

After convergence, run `fold.py` to collapse omega -> hard 1/|group| weights
and produce a standard VHA checkpoint (n_kv_heads=2 + postmix).

Distributed: pure DataParallel across N GPUs (no TP/PP/sharding).
Reuses dataloader/distributed-init/ckpt-load patterns from joint_refine.py.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time
from datetime import datetime
import numpy as np


# Same key remapping convention as joint_refine.py / cascading scripts
def ckpt_key_to_pipeline_name(key):
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


def main():
    # First-pass parser: only --config; then load JSON as defaults.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None,
                           help="JSON config file. CLI flags override JSON values.")
    boot_args, remaining = bootstrap.parse_known_args()
    json_defaults = {}
    if boot_args.config:
        with open(boot_args.config) as f:
            json_defaults = json.load(f)

    parser = argparse.ArgumentParser(parents=[bootstrap])
    # Required paths
    parser.add_argument("--gqa_checkpoint", type=str,
                        default=json_defaults.get("gqa_checkpoint"),
                        required="gqa_checkpoint" not in json_defaults,
                        help="HF safetensors dir of GQA pretrain ckpt-24000.")
    parser.add_argument("--gqa_model_config", type=str,
                        default=json_defaults.get("gqa_model_config"),
                        required="gqa_model_config" not in json_defaults,
                        help="GQA model config (provider name) used by qwen_provider.create_provider.")
    parser.add_argument("--groupings_path", type=str,
                        default=json_defaults.get("groupings_path"),
                        required="groupings_path" not in json_defaults,
                        help="Path to conversion_diagnostics.json with per-layer groupings.")
    parser.add_argument("--output_path", type=str,
                        default=json_defaults.get("output_path"),
                        required="output_path" not in json_defaults)
    # Data
    parser.add_argument("--data_path", type=str, default=json_defaults.get("data_path"))
    parser.add_argument("--num_samples", type=int, default=json_defaults.get("num_samples", 65536))
    parser.add_argument("--seq_length", type=int, default=json_defaults.get("seq_length", 4096),
                        help="Match GQA pretrain (4096) for distribution alignment.")
    # Training schedule
    parser.add_argument("--max_steps", type=int, default=json_defaults.get("max_steps", 2000))
    parser.add_argument("--lr", type=float, default=json_defaults.get("lr", 1e-5),
                        help="Constant LR; default=1e-5 = GQA pretrain min_lr (cosine end).")
    parser.add_argument("--grad_clip", type=float, default=json_defaults.get("grad_clip", 1.0))
    parser.add_argument("--weight_decay", type=float, default=json_defaults.get("weight_decay", 0.0))
    parser.add_argument("--adam_beta2", type=float, default=json_defaults.get("adam_beta2", 0.95))
    parser.add_argument("--batch_size", type=int, default=json_defaults.get("batch_size", 4),
                        help="Per-GPU batch size; matches GQA per_device_train_batch_size.")
    parser.add_argument("--grad_accum", type=int, default=json_defaults.get("grad_accum", 1))
    # ALM schedule
    parser.add_argument("--fusion_decay_steps", type=int,
                        default=json_defaults.get("fusion_decay_steps", 1500),
                        help="Linear decay of t(s) from t_0 to 0 over this many steps. "
                             "Last (max_steps - fusion_decay_steps) steps run at t=0 (hard constraint).")
    parser.add_argument("--lambda_init", type=float, default=json_defaults.get("lambda_init", 0.0))
    parser.add_argument("--lambda_lr", type=float, default=json_defaults.get("lambda_lr", 1.0))
    parser.add_argument("--lambda_max", type=float, default=json_defaults.get("lambda_max", 100.0))
    parser.add_argument("--dual_update_interval", type=int,
                        default=json_defaults.get("dual_update_interval", 50))
    # Logging / save
    parser.add_argument("--log_interval", type=int, default=json_defaults.get("log_interval", 20))
    parser.add_argument("--gate_check_interval", type=int,
                        default=json_defaults.get("gate_check_interval", 200),
                        help="Run 4-gate health check every N steps.")
    parser.add_argument("--save_interval", type=int, default=json_defaults.get("save_interval", 500),
                        help="Save fusion-mode ckpt every N steps (0 disables).")
    # Param freezing
    parser.add_argument("--train_mode", choices=["all", "attn_and_fusion", "fusion_only"],
                        default=json_defaults.get("train_mode", "all"),
                        help="all: train every param; attn_and_fusion: only attn proj + fusion; "
                             "fusion_only: only omega + postmix.")
    args = parser.parse_args()

    # ---- Distributed init (verbatim from joint_refine.py) ----
    import paddle
    import paddle.distributed as dist
    import paddle.distributed.fleet as fleet
    import paddlefleet
    import paddlefleet.tensor_parallel.random as _rng_module
    import paddlefleet.parallel_state as _ps

    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": world_size, "mp_degree": 1, "pp_degree": 1, "sharding_degree": 1,
    }
    fleet.fleet._user_defined_strategy = strategy

    _rng_module.initialize_rng_tracker()
    _rng_module._CUDA_RNG_STATE_TRACKER.fork = lambda name="model-parallel-rng": contextlib.nullcontext()
    _ps.get_tensor_model_parallel_rank = lambda: 0
    _ps.get_tensor_model_parallel_world_size = lambda: 1
    _ps.get_pipeline_model_parallel_rank = lambda: 0
    _ps.get_pipeline_model_parallel_world_size = lambda: 1
    _ps.get_data_parallel_rank = lambda: rank
    _ps.get_data_parallel_world_size = lambda: world_size
    _ps.get_expert_model_parallel_rank = lambda: 0
    _ps.get_expert_tensor_parallel_rank = lambda: 0
    _ps.get_expert_tensor_and_model_parallel_rank = lambda: 0

    from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    def _simple_linear(input_, weight, bias=None):
        out_shape = list(input_.shape[:-1]) + [weight.shape[-1]]
        flat_input = input_.reshape([-1, input_.shape[-1]]).cast("float32")
        out = paddle.matmul(flat_input, weight.cast("float32")).cast(input_.dtype).reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out

    def _simple_col_fwd(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else self.weight
        bias = self.bias if not self.skip_bias_add and self.bias is not None else None
        out = _simple_linear(input_, w, bias)
        output_bias = self.bias if self.skip_bias_add and self.bias is not None else None
        return out, output_bias

    def _simple_row_fwd(self, input_, weight=None):
        w = weight if weight is not None else self.weight
        bias = self.bias if not self.skip_bias_add and self.bias is not None else None
        out = _simple_linear(input_, w, bias)
        output_bias = self.bias if self.skip_bias_add and self.bias is not None else None
        return out, output_bias

    ColumnParallelLinear.forward = _simple_col_fwd
    RowParallelLinear.forward = _simple_row_fwd

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "VHA"))
    from models.qwen_provider import create_provider
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    # DHA fusion modules (from same dir as this script)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from grouping import load_groupings, validate_groupings
    from attention_patch import (
        install_all_fusion_hooks, collect_fusion_params, reset_caches,
    )
    from alm_loss import (
        ALMConfig, ALMState, compute_total_constraint, alm_combine,
    )

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    from safetensors import safe_open
    import ml_dtypes

    def load_ckpt_into(model, ckpt_dir):
        param_dict = dict(model.named_parameters())
        sf_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")])
        loaded = 0
        shape_mismatch = []
        for sf in sf_files:
            with safe_open(os.path.join(ckpt_dir, sf), framework="pt", device="cpu") as f:
                for key in f.keys():
                    val = f.get_tensor(key).float().numpy()
                    pname = ckpt_key_to_pipeline_name(key)
                    param = param_dict.get(pname) if pname else None
                    if param is None:
                        for cand in [key, "model." + key, key.replace("model.", "")]:
                            param = param_dict.get(cand)
                            if param is not None:
                                break
                    if param is not None:
                        if list(param.shape) == list(val.shape):
                            param.set_value(paddle.to_tensor(val).cast(param.dtype))
                            loaded += 1
                        else:
                            shape_mismatch.append((key, list(val.shape), list(param.shape)))
        if shape_mismatch:
            log(f"  [WARN] {len(shape_mismatch)} shape-mismatch params:")
            for k, vs, ps in shape_mismatch[:10]:
                log(f"    {k}: ckpt={vs} model={ps}")
            raise RuntimeError(f"Refusing to continue with {len(shape_mismatch)} silent skips")
        return loaded, len(param_dict)

    # ---- Load GQA model (will keep K_proj=8 heads throughout fusion) ----
    log("=" * 60)
    log("DHA-style fusion training: GQA(8KV) -> VHA(2KV+postmix)")
    log(f"  world_size={world_size}, rank={rank}")
    log("=" * 60)
    provider = create_provider(args.gqa_model_config)
    provider.seq_length = args.seq_length
    provider.max_sequence_length = args.seq_length
    model = provider.provide()
    n, t = load_ckpt_into(model, args.gqa_checkpoint)
    log(f"  Loaded {n}/{t} GQA params from {args.gqa_checkpoint}")

    # Locate transformer layers
    transformer_layers = [fn for fn in model.run_function if isinstance(fn, TransformerLayer)]
    num_layers = len(transformer_layers)
    log(f"  num_transformer_layers={num_layers}")

    # ---- Load groupings + install fusion hooks ----
    groupings = load_groupings(args.groupings_path)
    validate_groupings(groupings, n_layers=num_layers, n_src_heads=8, n_groups=2)
    log(f"  Loaded groupings for {num_layers} layers from {args.groupings_path}")
    fusion_states = install_all_fusion_hooks(
        transformer_layers,
        groupings,
        attention_attr="self_attn",
        core_attention_attr="core_attention",
        kv_arg_indices=(1, 2),
        n_groups=2, total_q_heads=16, head_dim=128, postmix_rank=4,
    )
    log(f"  Installed {len(fusion_states)} fusion hooks")
    fusion_param_count = sum(int(np.prod(p.shape)) for fs in fusion_states for p in fs.parameters())
    log(f"  Fusion params: {fusion_param_count} elements across {len(fusion_states)} layers")

    # ---- Param freezing ----
    train_params = []
    if args.train_mode == "fusion_only":
        for p in model.parameters():
            p.stop_gradient = True
        for fs in fusion_states:
            for p in fs.parameters():
                p.stop_gradient = False
                train_params.append(p)
    elif args.train_mode == "attn_and_fusion":
        for p in model.parameters():
            p.stop_gradient = True
        for fn in transformer_layers:
            for name, p in fn.named_parameters():
                if "self_attn" in name:
                    p.stop_gradient = False
                    train_params.append(p)
        # fusion params are children of self_attn.dha_fusion -> already covered above,
        # but de-dup just in case
        seen = set(id(p) for p in train_params)
        for fs in fusion_states:
            for p in fs.parameters():
                if id(p) not in seen:
                    p.stop_gradient = False
                    train_params.append(p)
                    seen.add(id(p))
    else:  # all
        for p in model.parameters():
            p.stop_gradient = False
            train_params.append(p)

    n_elem = sum(int(np.prod(p.shape)) for p in train_params)
    log(f"  Training {len(train_params)} params, {n_elem} elements ({args.train_mode})")

    # ---- Load tokens (same MMap shard pattern as joint_refine.py) ----
    if args.data_path:
        from paddleformers.data.indexed_dataset import MMapIndexedDataset
        mmap_ds = MMapIndexedDataset(args.data_path, skip_warmup=True)
        token_ids = []
        idx = 0
        while len(token_ids) < args.num_samples and idx < len(mmap_ds):
            toks = mmap_ds[idx]
            if len(toks) >= args.seq_length:
                token_ids.append(toks[: args.seq_length].astype(np.int64))
            idx += 1
        token_ids = np.stack(token_ids, axis=0)
    else:
        token_ids = np.random.randint(100, 151000, (args.num_samples, args.seq_length))
    N_total = token_ids.shape[0]
    shard_size = N_total // world_size
    shard_start = rank * shard_size
    shard_end = shard_start + shard_size if rank < world_size - 1 else N_total
    token_ids = token_ids[shard_start:shard_end]
    N_local = token_ids.shape[0]
    log(f"  Total {N_total} samples, {N_local} per GPU")

    # ---- Optimizer + ALM state ----
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.lr, beta1=0.9, beta2=args.adam_beta2,
        epsilon=1e-8, parameters=train_params, weight_decay=args.weight_decay,
    )
    alm_cfg = ALMConfig(
        target_initial=-1.0,                     # auto-tune to first measured C
        target_decay_steps=args.fusion_decay_steps,
        auto_tune_initial=True,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        lambda_max=args.lambda_max,
        dual_update_interval=args.dual_update_interval,
    )
    alm = ALMState(alm_cfg)

    # ---- LM loss helper ----
    def forward_lm(input_ids):
        x = {"input_ids": input_ids}
        for fn in model.run_function:
            x = fn(x)
        if isinstance(x, dict):
            for key in ("logits", "output"):
                if key in x:
                    return x[key]
            raise RuntimeError(f"forward: cannot find logits in {list(x.keys())}")
        return x

    def lm_loss_from_logits(logits, input_ids):
        # Standard causal LM loss: shift labels right
        shift_logits = logits[:, :-1, :].cast("float32")
        shift_labels = input_ids[:, 1:]
        loss = paddle.nn.functional.cross_entropy(
            shift_logits.reshape([-1, shift_logits.shape[-1]]),
            shift_labels.reshape([-1]),
            reduction="mean",
        )
        return loss

    # ---- Save helper ----
    def save_fusion_ckpt(out_dir, tag=""):
        if rank != 0:
            return
        os.makedirs(out_dir, exist_ok=True)
        from safetensors.numpy import save_file
        output_tensors = {}
        for name, param in model.named_parameters():
            val = param.cast("float32").numpy().astype(ml_dtypes.bfloat16)
            parts = name.split(".", 1)
            try:
                idx = int(parts[0])
                rest = parts[1]
                if idx == 0:
                    out_key = f"model.{rest}"
                elif idx == 29:
                    out_key = "model.norm.weight"
                elif idx == 30:
                    out_key = "model.lm_head.weight"
                else:
                    out_key = f"model.layers.{idx - 1}.{rest}"
            except (ValueError, IndexError):
                out_key = name
            output_tensors[out_key] = val
        if ("model.lm_head.weight" not in output_tensors
                and "model.embedding.embed_tokens.weight" in output_tensors):
            output_tensors["model.lm_head.weight"] = output_tensors[
                "model.embedding.embed_tokens.weight"]
        out_file = os.path.join(out_dir, "model-00001-of-00001.safetensors")
        save_file(output_tensors, out_file)
        weight_map = {k: "model-00001-of-00001.safetensors" for k in output_tensors.keys()}
        total_size = sum(t.nbytes for t in output_tensors.values())
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=4)
        # ALM state for resumption / postmortem
        alm_state_path = os.path.join(out_dir, "alm_state.json")
        with open(alm_state_path, "w") as f:
            json.dump({
                "step": alm.step,
                "lam": alm.lam,
                "target_initial": alm.target_initial,
                "current_target": alm.current_target(),
                "history_tail": alm.history[-50:],
                "tag": tag,
            }, f, indent=2)
        # Copy GQA config so fold.py knows starting architecture
        import shutil
        src_config = os.path.join(args.gqa_checkpoint, "config.json")
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(out_dir, "config.json"))
        # Save fusion-mode marker + groupings for fold.py
        fusion_meta = {
            "fusion_mode": True,
            "groupings": groupings,
            "n_groups": 2,
            "total_q_heads": 16,
            "head_dim": 128,
            "postmix_rank": 4,
        }
        with open(os.path.join(out_dir, "dha_fusion_meta.json"), "w") as f:
            json.dump(fusion_meta, f, indent=2)

    # ---- Train ----
    log("\n" + "=" * 60)
    log("Training")
    log(f"  lr={args.lr} (constant), max_steps={args.max_steps}")
    log(f"  ALM: t_0=auto, decay_steps={args.fusion_decay_steps}, "
        f"lambda_init={args.lambda_init}, lambda_lr={args.lambda_lr}, lambda_max={args.lambda_max}")
    log(f"  gates every {args.gate_check_interval} steps; save every {args.save_interval} steps")
    log("=" * 60)

    mini_batch = min(args.batch_size, N_local)
    initial_lm = None
    train_start_time = time.time()
    last_log_time = train_start_time
    last_log_step = 0
    tokens_per_step = world_size * args.batch_size * args.seq_length * args.grad_accum

    for step in range(args.max_steps):
        accum_lm = 0.0
        accum_constraint = 0.0
        accum_total = 0.0

        # Stash loss tensors; defer .item() to syncs needed by dual-update / logging.
        lm_t_acc = None
        constraint_t_acc = None
        total_t_acc = None

        for _ in range(args.grad_accum):
            indices = np.random.choice(N_local, mini_batch, replace=False)
            batch = token_ids[indices]
            input_ids = paddle.to_tensor(batch.astype(np.int64))

            reset_caches(fusion_states)
            logits = forward_lm(input_ids)
            lm = lm_loss_from_logits(logits, input_ids)
            constraint = compute_total_constraint(fusion_states)
            total = alm_combine(lm, constraint, alm)
            scaled = total / args.grad_accum
            scaled.backward()

            lm_d = lm.detach()
            c_d = constraint.detach()
            tot_d = total.detach()
            lm_t_acc = lm_d if lm_t_acc is None else (lm_t_acc + lm_d)
            constraint_t_acc = c_d if constraint_t_acc is None else (constraint_t_acc + c_d)
            total_t_acc = tot_d if total_t_acc is None else (total_t_acc + tot_d)

        # Average over grad_accum (still tensor; no sync yet)
        inv_ga = 1.0 / args.grad_accum
        lm_t = lm_t_acc * inv_ga
        constraint_t = constraint_t_acc * inv_ga
        total_t = total_t_acc * inv_ga

        # All-reduce metrics as a single 3-vec (still tensor; no .item() yet)
        if world_size > 1:
            metrics = paddle.stack([lm_t.cast("float32"), constraint_t.cast("float32"), total_t.cast("float32")])
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics = metrics * (1.0 / world_size)
            lm_t, constraint_t, total_t = metrics[0], metrics[1], metrics[2]

        # All-reduce gradients (FUSED: one NCCL call per dtype, not per parameter)
        if world_size > 1:
            inv_ws = 1.0 / world_size
            from collections import defaultdict
            buckets = defaultdict(list)
            for p in train_params:
                if p.grad is not None:
                    buckets[str(p.grad.dtype)].append(p)
            for _, plist in buckets.items():
                shapes = [tuple(p.grad.shape) for p in plist]
                numels = [int(np.prod(s)) for s in shapes]
                flat = paddle.concat([p.grad.reshape([-1]) for p in plist])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat = flat * inv_ws
                offset = 0
                for p, n, s in zip(plist, numels, shapes):
                    p.grad = flat[offset:offset + n].reshape(list(s))
                    offset += n

        # Grad clip (no fp32 cast over all params; native dtype square then small fp32 partial sums)
        if args.grad_clip > 0:
            partials = []
            for p in train_params:
                if p.grad is not None:
                    # sum-of-squares stays in source dtype; cast scalar to fp32 only.
                    s = paddle.sum(paddle.square(p.grad)).cast("float32")
                    partials.append(s)
            if partials:
                gn_sq = paddle.add_n(partials)
                gn = paddle.sqrt(gn_sq)
                clip_val = paddle.full([], args.grad_clip, dtype="float32")
                coef = paddle.minimum(clip_val / (gn + 1e-6), paddle.ones_like(clip_val))
                for p in train_params:
                    if p.grad is not None:
                        p.grad = p.grad * coef.cast(p.grad.dtype)

        optimizer.step()
        optimizer.clear_grad()

        # Decide whether this step needs scalar values (sync)
        is_log_step = (rank == 0 and (step % args.log_interval == 0 or step == args.max_steps - 1))
        is_gate_step = (rank == 0 and step > 0 and step % args.gate_check_interval == 0)
        is_dual_step = (step > 0 and step % args.dual_update_interval == 0)
        need_scalars = is_log_step or is_gate_step or is_dual_step or (initial_lm is None)

        if need_scalars:
            accum_lm = float(lm_t.item())
            accum_constraint = float(constraint_t.item())
            accum_total = float(total_t.item())
            # ALM dual update on the averaged constraint (only on dual steps)
            if is_dual_step:
                alm.maybe_update_lambda(accum_constraint)
            # postmix V norm summary (defer expensive sync to log/gate steps)
            if is_log_step or is_gate_step:
                pm_sq = None
                for fs in fusion_states:
                    s = paddle.sum(paddle.square(fs.postmix_V.cast("float32")))
                    pm_sq = s if pm_sq is None else pm_sq + s
                postmix_norm = float(paddle.sqrt(pm_sq / max(1, len(fusion_states))).item())
            else:
                postmix_norm = 0.0
            alm.end_step(
                accum_constraint, accum_lm,
                log_extra={"postmix_V_avg_norm": postmix_norm} if postmix_norm > 0 else None,
            )
        else:
            # No-sync path: bookkeep ALM step counter without recording history this step
            alm.step += 1

        if initial_lm is None:
            initial_lm = accum_lm

        # ---- Logging ----
        if is_log_step:
            now = time.time()
            elapsed_total = now - train_start_time
            steps_in_window = max(1, step - last_log_step)
            window_elapsed = max(1e-6, now - last_log_time)
            sec_per_step = window_elapsed / steps_in_window
            tok_per_sec = tokens_per_step / sec_per_step
            remaining_steps = args.max_steps - step
            eta_sec = remaining_steps * sec_per_step
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] step {step:4d}: lm={accum_lm:.4f} (init {initial_lm:.4f}) "
                f"C={accum_constraint:.5f} t={alm.current_target():.5f} "
                f"lam={alm.lam:.3f} total={accum_total:.4f} "
                f"postmix_V_norm={postmix_norm:.4f} "
                f"| {sec_per_step:.2f}s/step {tok_per_sec/1000:.1f}Ktok/s "
                f"elapsed={elapsed_total/60:.1f}min ETA={eta_sec/60:.1f}min",
                flush=True,
            )
            last_log_time = now
            last_log_step = step

        # ---- 4-gate health check ----
        if rank == 0 and step > 0 and step % args.gate_check_interval == 0:
            gate1 = accum_constraint
            gate2 = accum_lm - initial_lm
            gate3 = postmix_norm
            gate_msgs = []
            if gate1 > alm.target_initial * 0.9 and step > args.fusion_decay_steps // 3:
                gate_msgs.append(f"GATE1: constraint={gate1:.4f} not decreasing")
            if gate2 > 0.5:
                gate_msgs.append(f"GATE2: LM loss drift={gate2:.4f} > 0.5")
            if step > args.fusion_decay_steps and gate3 < 1e-3:
                gate_msgs.append(f"GATE3: postmix V norm={gate3:.5f} too small late in training")
            if gate_msgs:
                print(f"  [GATES@{step}] WARN: " + "; ".join(gate_msgs), flush=True)
            else:
                print(f"  [GATES@{step}] OK (C={gate1:.4f} dLM={gate2:+.4f} pm={gate3:.4f})",
                      flush=True)

        # ---- Periodic save ----
        if (args.save_interval > 0 and step > 0 and step % args.save_interval == 0):
            save_dir = os.path.join(args.output_path, f"step_{step}")
            save_fusion_ckpt(save_dir, tag=f"step_{step}")
            log(f"  [saved fusion ckpt to {save_dir}]")

    # ---- Final save ----
    log(f"\nDone. Final state: step={alm.step} C={alm.history[-1]['constraint']:.5f} "
        f"lam={alm.lam:.3f}")
    if rank == 0:
        save_fusion_ckpt(args.output_path, tag="final")
        with open(os.path.join(args.output_path, "alm_history.json"), "w") as f:
            json.dump(alm.history, f, indent=2)
        log(f"\nFinal fusion ckpt saved to {args.output_path}")
        log("Next step: run fold.py to collapse omega -> 2-head VHA ckpt.")


if __name__ == "__main__":
    main()
