#!/usr/bin/env python3
"""Joint logit-level refinement for VHA checkpoint.

Loads a refined VHA student + GQA teacher, then trains all student attention
parameters jointly with KL(student || teacher) loss on logits. Used as a
final-stage refinement after cascading block-MSE refine to bridge the gap
from per-layer hidden alignment to logit-level distribution.

Distributed: data parallelism across N GPUs, gradient all-reduce.
"""

import argparse
import contextlib
import gc
import json
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
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--refine_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_min_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--train_mode", choices=["attn", "attn_norm", "full"], default="attn")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save intermediate checkpoint every N steps (0 to disable).")
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
                f"Refusing to continue: {len(shape_mismatch)} params silently skipped due to shape mismatch. "
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

    warmup_steps = max(1, int(args.refine_steps * args.warmup_ratio))
    min_lr = args.lr * args.lr_min_ratio

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.refine_steps - warmup_steps)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (args.lr - min_lr) * cosine

    def forward_logits(model, input_ids):
        x = {"input_ids": input_ids}
        for fn in model.run_function:
            x = fn(x)
        # The final lm_head returns a Tensor directly; intermediate stages
        # return dicts.
        if isinstance(x, dict):
            for key in ("logits", "output", "hidden_states"):
                if key in x:
                    return x[key]
            raise RuntimeError(f"forward_logits: cannot find logits in {list(x.keys())}")
        return x

    # ---- Train ----
    log("\n" + "=" * 60)
    log("Joint logit refine")
    log("=" * 60)
    log(f"  lr={args.lr}, steps={args.refine_steps}, warmup={warmup_steps}, "
        f"T={args.kl_temperature}, batch={args.batch_size}, accum={args.grad_accum}")

    mini_batch = min(args.batch_size, N_local)
    T = args.kl_temperature
    initial_loss = None
    best_loss = float('inf')

    for step in range(args.refine_steps):
        cur_lr = get_lr(step)
        optimizer.set_lr(cur_lr)

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            indices = np.random.choice(N_local, mini_batch, replace=False)
            batch = token_ids[indices]
            input_ids = paddle.to_tensor(batch.astype(np.int64))

            with paddle.no_grad():
                t_logits = forward_logits(gqa_model, input_ids).cast("float32")
                t_logp = paddle.nn.functional.log_softmax(t_logits / T, axis=-1)
                t_p = paddle.exp(t_logp)

            s_logits = forward_logits(vha_model, input_ids).cast("float32")
            s_logp = paddle.nn.functional.log_softmax(s_logits / T, axis=-1)
            # KL(teacher || student) = sum t_p * (t_logp - s_logp)
            kl = (t_p * (t_logp - s_logp)).sum(axis=-1).mean() * (T * T)

            scaled = kl / args.grad_accum
            scaled.backward()
            accum_loss += float(kl.item())

        accum_loss /= args.grad_accum

        # all-reduce loss for consistent logging/decisions
        if world_size > 1:
            _l = paddle.to_tensor([accum_loss], dtype="float32")
            dist.all_reduce(_l, op=dist.ReduceOp.SUM)
            accum_loss = float(_l.item()) / world_size

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

        if initial_loss is None:
            initial_loss = accum_loss
        if accum_loss < best_loss:
            best_loss = accum_loss

        if rank == 0 and (step % args.log_interval == 0 or step == args.refine_steps - 1):
            impr = (1 - accum_loss / initial_loss) * 100 if initial_loss > 0 else 0
            print(f"  step {step}: KL={accum_loss:.4f} (best={best_loss:.4f}, "
                  f"{impr:.1f}% reduction) lr={cur_lr:.2e}", flush=True)

        # Optional intermediate save
        if (args.save_interval > 0 and rank == 0
                and step > 0 and step % args.save_interval == 0):
            save_dir = os.path.join(args.output_path, f"step_{step}")
            _save(vha_model, save_dir, args.vha_checkpoint, ml_dtypes)
            print(f"  [saved intermediate to {save_dir}]", flush=True)

    # broadcast final params from rank 0 to be safe
    if world_size > 1:
        for p in train_params:
            dist.broadcast(p, src=0)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        _save(vha_model, args.output_path, args.vha_checkpoint, ml_dtypes)
        print(f"\nDone! Joint logit-refined ckpt saved to {args.output_path}", flush=True)


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
