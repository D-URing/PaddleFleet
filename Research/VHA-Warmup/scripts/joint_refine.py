#!/usr/bin/env python3
"""Unified joint refine for VHA: block_MSE + logit_KL + final_hidden_MSE.

Single-stage replacement for the sequential cascade(MSE) -> joint_logit(KL)
pipeline. Trains all student attention params jointly with a combined loss,
using plateau-based termination instead of fixed step count, motivated by
DHA's "L_fusion < 1e-3 then stop" recipe.

Loss = alpha * mean_layer(block_MSE) + beta * logit_KL + gamma * final_hidden_MSE
alpha/beta/gamma can be annealed: early steps emphasize local block alignment,
later steps emphasize global logit/hidden alignment.

Plateau termination: stop when rolling-best over `plateau_window` improves
by less than `plateau_eps` for `plateau_window` consecutive steps, after a
minimum of `min_steps`.

Distributed: data parallel across N GPUs, gradient + loss all-reduce.
"""

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import sys
import numpy as np


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_checkpoint", type=str, required=True)
    parser.add_argument("--gqa_model_config", type=str, required=True)
    parser.add_argument("--vha_checkpoint", type=str, required=True)
    parser.add_argument("--vha_model_config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=65536,
                        help="Total samples across all ranks. Each rank loads only its shard.")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="Hard cap on steps; plateau detection usually triggers earlier.")
    parser.add_argument("--min_steps", type=int, default=500,
                        help="Minimum steps before plateau termination is allowed.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--lr_min_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--train_mode", choices=["attn", "attn_norm", "full"], default="attn")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save intermediate ckpt every N steps (0 disables).")
    # Loss weight schedule
    parser.add_argument("--alpha_start", type=float, default=1.0,
                        help="Initial weight on block_MSE (mean over layers).")
    parser.add_argument("--alpha_end", type=float, default=0.1)
    parser.add_argument("--beta_start", type=float, default=0.1,
                        help="Initial weight on logit_KL.")
    parser.add_argument("--beta_end", type=float, default=1.0)
    parser.add_argument("--gamma_start", type=float, default=0.0,
                        help="Initial weight on final_hidden_MSE.")
    parser.add_argument("--gamma_end", type=float, default=0.1)
    parser.add_argument("--anneal_steps", type=int, default=2000,
                        help="Steps over which to linearly interpolate alpha/beta/gamma.")
    # Plateau detection
    parser.add_argument("--plateau_window", type=int, default=300,
                        help="Window size (steps) for rolling-best plateau detection.")
    parser.add_argument("--plateau_eps", type=float, default=0.005,
                        help="Relative improvement threshold; below this the loss is considered plateaued.")
    parser.add_argument("--plateau_metric", choices=["total", "kl"], default="kl",
                        help="Which loss to track for plateau detection.")
    args = parser.parse_args()

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

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'VHA'))
    from models.qwen_provider import create_provider
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    from safetensors import safe_open
    import ml_dtypes

    def load_ckpt_into(model, ckpt_dir):
        param_dict = dict(model.named_parameters())
        sf_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')])
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
            log(f"  [WARN] {len(shape_mismatch)} shape-mismatch params (skipped):")
            for k, vs, ps in shape_mismatch[:10]:
                log(f"    {k}: ckpt={vs} model={ps}")
            raise RuntimeError(
                f"Refusing to continue: {len(shape_mismatch)} params silently skipped. "
                f"Likely VHA_MODEL_CONFIG architecture does not match VHA_CHECKPOINT.")
        return loaded, len(param_dict)

    # ---- Load teacher (GQA) ----
    log("=" * 60)
    log("Loading GQA teacher")
    log(f"  world_size={world_size}, rank={rank}")
    log("=" * 60)
    provider = create_provider(args.gqa_model_config)
    provider.seq_length = args.seq_length
    provider.max_sequence_length = args.seq_length
    gqa_model = provider.provide()
    n, t = load_ckpt_into(gqa_model, args.gqa_checkpoint)
    log(f"  Loaded {n}/{t} GQA params")
    gqa_model.eval()
    for p in gqa_model.parameters():
        p.stop_gradient = True

    # ---- Load student (VHA) ----
    log("\nLoading VHA student")
    vha_provider = create_provider(args.vha_model_config)
    vha_provider.seq_length = args.seq_length
    vha_provider.max_sequence_length = args.seq_length
    vha_model = vha_provider.provide()
    n, t = load_ckpt_into(vha_model, args.vha_checkpoint)
    log(f"  Loaded {n}/{t} VHA params")

    # Freeze all, then unfreeze attn params per train_mode
    for p in vha_model.parameters():
        p.stop_gradient = True

    train_params = []
    for fn in vha_model.run_function:
        if isinstance(fn, TransformerLayer):
            for name, p in fn.named_parameters():
                want = False
                if args.train_mode == "full":
                    want = True
                elif args.train_mode == "attn":
                    want = "self_attention" in name or "self_attn" in name
                elif args.train_mode == "attn_norm":
                    want = ("self_attention" in name or "self_attn" in name
                            or "input_norm" in name or "post_attention_norm" in name)
                if want:
                    p.stop_gradient = False
                    train_params.append(p)
    n_elem = sum(int(np.prod(p.shape)) for p in train_params)
    log(f"  Training {len(train_params)} params, {n_elem} elements ({args.train_mode})")

    # Locate TransformerLayer indices for both models, plus terminal stages
    gqa_tl = [i for i, fn in enumerate(gqa_model.run_function) if isinstance(fn, TransformerLayer)]
    vha_tl = [i for i, fn in enumerate(vha_model.run_function) if isinstance(fn, TransformerLayer)]
    if len(gqa_tl) != len(vha_tl):
        raise RuntimeError(f"Layer count mismatch: GQA={len(gqa_tl)} VHA={len(vha_tl)}")
    num_layers = len(gqa_tl)
    log(f"  num_layers={num_layers}")

    # ---- Load tokens ----
    if args.data_path:
        from paddleformers.data.indexed_dataset import MMapIndexedDataset
        mmap_ds = MMapIndexedDataset(args.data_path, skip_warmup=True)
        token_ids = []
        idx = 0
        while len(token_ids) < args.num_samples and idx < len(mmap_ds):
            toks = mmap_ds[idx]
            if len(toks) >= args.seq_length:
                token_ids.append(toks[:args.seq_length].astype(np.int64))
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

    # ---- Optimizer ----
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.lr, beta1=0.9, beta2=args.adam_beta2,
        epsilon=1e-8, parameters=train_params, weight_decay=args.weight_decay,
    )

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        min_lr = args.lr * args.lr_min_ratio
        return min_lr + (args.lr - min_lr) * cosine

    def get_weights(step):
        # Linear interpolation over anneal_steps
        t_frac = min(1.0, step / max(args.anneal_steps, 1))
        a = args.alpha_start + (args.alpha_end - args.alpha_start) * t_frac
        b = args.beta_start + (args.beta_end - args.beta_start) * t_frac
        g = args.gamma_start + (args.gamma_end - args.gamma_start) * t_frac
        return a, b, g

    def forward_with_blocks(model, tl_indices, input_ids, want_grad):
        """Run model layer-by-layer, returning (per_layer_hidden_list, final_logits).

        per_layer_hidden_list[i] is the hidden_states OUTPUT of TransformerLayer i.
        Pre-layer stages (e.g. embedding) and post-layer stages (norm, lm_head)
        are run as usual.
        """
        ctx = contextlib.nullcontext() if want_grad else paddle.no_grad()
        with ctx:
            x = {"input_ids": input_ids}
            tl_set = set(tl_indices)
            per_layer = []
            for i, fn in enumerate(model.run_function):
                x = fn(x)
                if i in tl_set:
                    # x is a dict here; capture hidden_states
                    per_layer.append(x["hidden_states"])
            # x at this point is the lm_head output (Tensor) or a dict containing logits
            if isinstance(x, dict):
                for key in ("logits", "output", "hidden_states"):
                    if key in x:
                        logits = x[key]
                        break
                else:
                    raise RuntimeError(f"forward: cannot find logits in {list(x.keys())}")
            else:
                logits = x
            return per_layer, logits

    # ---- Train ----
    log("\n" + "=" * 60)
    log("Joint refine (block_MSE + logit_KL + final_hidden_MSE)")
    log("=" * 60)
    log(f"  lr={args.lr}, max_steps={args.max_steps}, warmup={args.warmup_steps}, "
        f"min_steps={args.min_steps}")
    log(f"  weights schedule: alpha {args.alpha_start}->{args.alpha_end}, "
        f"beta {args.beta_start}->{args.beta_end}, gamma {args.gamma_start}->{args.gamma_end} "
        f"over {args.anneal_steps} steps")
    log(f"  plateau: window={args.plateau_window}, eps={args.plateau_eps}, "
        f"metric={args.plateau_metric}")

    mini_batch = min(args.batch_size, N_local)
    T = args.kl_temperature

    # Plateau detection state: track best metric in the most recent window.
    history = []  # list of (step, metric)
    best_total = float('inf')
    best_kl = float('inf')
    best_param_states = None
    best_step = -1
    last_snapshot_metric = float('inf')
    last_snapshot_step = -10**9

    initial_total = None
    stop_reason = None

    for step in range(args.max_steps):
        cur_lr = get_lr(step)
        optimizer.set_lr(cur_lr)
        alpha, beta, gamma = get_weights(step)

        accum_total = 0.0
        accum_block = 0.0
        accum_kl = 0.0
        accum_fh = 0.0

        for _ in range(args.grad_accum):
            indices = np.random.choice(N_local, mini_batch, replace=False)
            batch = token_ids[indices]
            input_ids = paddle.to_tensor(batch.astype(np.int64))

            # Teacher: no_grad, capture per-layer hiddens + logits
            t_hids, t_logits = forward_with_blocks(gqa_model, gqa_tl, input_ids, want_grad=False)
            # Detach to be safe; cast to fp32 for stable loss
            t_hids = [h.detach().cast("float32") for h in t_hids]
            t_logits_f = t_logits.cast("float32")
            t_logp = paddle.nn.functional.log_softmax(t_logits_f / T, axis=-1)
            t_p = paddle.exp(t_logp)
            # Detach KL targets
            t_logp = t_logp.detach()
            t_p = t_p.detach()

            # Student: grad on
            s_hids, s_logits = forward_with_blocks(vha_model, vha_tl, input_ids, want_grad=True)

            # Block MSE: mean over layers of MSE(s_hid_l, t_hid_l)
            block_loss = None
            for sh, th in zip(s_hids, t_hids):
                m = paddle.nn.functional.mse_loss(sh.cast("float32"), th)
                block_loss = m if block_loss is None else block_loss + m
            block_loss = block_loss / num_layers

            # Final hidden MSE (last layer's output): redundant with last block_MSE term
            # but kept as a separate weighted signal so the schedule can emphasize it.
            final_hidden_loss = paddle.nn.functional.mse_loss(
                s_hids[-1].cast("float32"), t_hids[-1])

            # Logit KL(teacher || student)
            s_logits_f = s_logits.cast("float32")
            s_logp = paddle.nn.functional.log_softmax(s_logits_f / T, axis=-1)
            kl = (t_p * (t_logp - s_logp)).sum(axis=-1).mean() * (T * T)

            total = alpha * block_loss + beta * kl + gamma * final_hidden_loss
            scaled = total / args.grad_accum
            scaled.backward()

            accum_total += float(total.item())
            accum_block += float(block_loss.item())
            accum_kl += float(kl.item())
            accum_fh += float(final_hidden_loss.item())

        accum_total /= args.grad_accum
        accum_block /= args.grad_accum
        accum_kl /= args.grad_accum
        accum_fh /= args.grad_accum

        # all-reduce metrics for consistent decisions across ranks
        if world_size > 1:
            metrics = paddle.to_tensor(
                [accum_total, accum_block, accum_kl, accum_fh], dtype="float32")
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics = metrics / world_size
            accum_total, accum_block, accum_kl, accum_fh = [float(x) for x in metrics.numpy()]

        # all-reduce gradients
        if world_size > 1:
            for p in train_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad = p.grad * (1.0 / world_size)

        # grad clip in fp32
        if args.grad_clip > 0:
            gn_sq = None
            for p in train_params:
                if p.grad is not None:
                    sq = paddle.sum(paddle.square(p.grad.cast("float32")))
                    gn_sq = sq if gn_sq is None else gn_sq + sq
            if gn_sq is not None:
                gn = paddle.sqrt(gn_sq)
                coef = args.grad_clip / (gn + 1e-6)
                if coef.item() < 1.0:
                    for p in train_params:
                        if p.grad is not None:
                            p.grad = p.grad * coef.cast(p.grad.dtype)

        optimizer.step()
        optimizer.clear_grad()

        if initial_total is None:
            initial_total = accum_total

        # Track best on selected plateau metric and snapshot params
        metric = accum_kl if args.plateau_metric == "kl" else accum_total
        if accum_total < best_total:
            best_total = accum_total
        if accum_kl < best_kl:
            best_kl = accum_kl
        cur_best_metric = best_kl if args.plateau_metric == "kl" else best_total
        if metric <= cur_best_metric + 1e-12:
            # New best — save snapshot (rank 0 only to save host memory; others can rebuild via broadcast at end)
            if rank == 0:
                best_param_states = [p.numpy().copy() for p in train_params]
            best_step = step

        history.append((step, metric))

        if rank == 0 and (step % args.log_interval == 0 or step == args.max_steps - 1):
            impr = (1 - accum_total / initial_total) * 100 if initial_total > 0 else 0
            print(f"  step {step}: total={accum_total:.4f} (block={accum_block:.5f} "
                  f"kl={accum_kl:.4f} fh={accum_fh:.5f}) "
                  f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f} "
                  f"best_kl={best_kl:.4f}@{best_step} "
                  f"lr={cur_lr:.2e} ({impr:.1f}% total reduction)", flush=True)

        # Plateau detection: compare best metric in last W steps vs best metric W steps ago.
        if step + 1 >= args.min_steps and step + 1 >= 2 * args.plateau_window:
            recent = [m for (s, m) in history[-args.plateau_window:]]
            prior = [m for (s, m) in history[-2 * args.plateau_window:-args.plateau_window]]
            best_recent = min(recent)
            best_prior = min(prior)
            if best_prior > 0:
                rel_improvement = (best_prior - best_recent) / best_prior
            else:
                rel_improvement = 0.0
            if rel_improvement < args.plateau_eps:
                stop_reason = (f"plateau: recent_best={best_recent:.5f} vs "
                               f"prior_best={best_prior:.5f}, rel_impr={rel_improvement:.4f} "
                               f"< eps={args.plateau_eps}")
                log(f"  [PLATEAU at step {step}] {stop_reason}")
                break

        # Optional intermediate save
        if (args.save_interval > 0 and rank == 0
                and step > 0 and step % args.save_interval == 0):
            save_dir = os.path.join(args.output_path, f"step_{step}")
            _save(vha_model, save_dir, args.vha_checkpoint, ml_dtypes)
            print(f"  [saved intermediate to {save_dir}]", flush=True)

    if stop_reason is None:
        stop_reason = f"max_steps={args.max_steps} reached"
    log(f"\nStopped: {stop_reason}")
    log(f"Best: total={best_total:.4f} kl={best_kl:.4f} at step {best_step}")

    # Roll back to best on rank 0, then broadcast to all ranks
    if rank == 0 and best_param_states is not None:
        import paddle as _pd
        for p, val in zip(train_params, best_param_states):
            p.set_value(_pd.to_tensor(val).cast(p.dtype))
        log(f"  Rolled back to best snapshot at step {best_step}")
    if world_size > 1:
        for p in train_params:
            dist.broadcast(p, src=0)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        _save(vha_model, args.output_path, args.vha_checkpoint, ml_dtypes)
        print(f"\nDone! Joint-refined ckpt saved to {args.output_path}", flush=True)


def _save(vha_model, out_dir, src_ckpt_dir, ml_dtypes):
    import paddle
    from safetensors.numpy import save_file
    os.makedirs(out_dir, exist_ok=True)
    output_tensors = {}
    for name, param in vha_model.named_parameters():
        val = param.cast("float32").numpy().astype(ml_dtypes.bfloat16)
        parts = name.split(".", 1)
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
        output_tensors[out_key] = val
    if ("model.lm_head.weight" not in output_tensors
            and "model.embedding.embed_tokens.weight" in output_tensors):
        output_tensors["model.lm_head.weight"] = output_tensors[
            "model.embedding.embed_tokens.weight"]
    output_file = os.path.join(out_dir, "model-00001-of-00001.safetensors")
    save_file(output_tensors, output_file)
    weight_map = {k: "model-00001-of-00001.safetensors" for k in output_tensors.keys()}
    total_size = sum(t.nbytes for t in output_tensors.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=4)
    src_config = os.path.join(src_ckpt_dir, "config.json")
    if os.path.exists(src_config):
        shutil.copy(src_config, os.path.join(out_dir, "config.json"))


if __name__ == "__main__":
    main()
