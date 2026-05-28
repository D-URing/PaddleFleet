#!/usr/bin/env python3
"""Distributed cascading refinement for VHA checkpoint.

Cascading approach with data parallelism across multiple GPUs:
  For layer N: input = VHA layers 0..N-1 output, target = GQA layer N output.
  Each GPU processes a shard of data, gradients are all-reduced.

Launch: python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 refine_vha_cascading.py ...
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
    """Convert checkpoint key to PipelineLayer parameter name."""
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
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to mmap dataset (basename without .bin/.idx)")
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--refine_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_decay_rate", type=float, default=0.85,
                        help="Per-layer lr multiplier: layer_lr = lr * decay^layer_idx")
    parser.add_argument("--lr_min_ratio", type=float, default=0.05,
                        help="Minimum lr ratio relative to layer lr (for cosine schedule)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup steps as fraction of refine_steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay (match training config)")
    parser.add_argument("--patience", type=int, default=120,
                        help="Early stop if no improvement for this many steps")
    parser.add_argument("--target_reduction", type=float, default=0.90,
                        help="Early stop once best loss reduction reaches this ratio after min_steps")
    parser.add_argument("--min_steps", type=int, default=200,
                        help="Minimum steps before target_reduction early stop")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps to increase effective batch size")
    parser.add_argument("--adam_beta1_decay", type=float, default=1.0,
                        help="Per-layer beta1 multiplier: beta1 = 0.9 * decay^layer_idx")
    parser.add_argument("--adam_beta2", type=float, default=0.95,
                        help="Adam beta2 (match training config)")
    parser.add_argument("--train_window", type=int, default=1,
                        help="Number of trailing VHA layers to train jointly for each target endpoint")
    parser.add_argument("--segment_stride", type=int, default=1,
                        help="Endpoint stride for segment refinement. 1 refines every layer; set equal to train_window for non-overlapping segments.")
    parser.add_argument("--train_mode", choices=["attn", "attn_norm", "full"], default="attn",
                        help="Parameters to train inside each window: self_attn only, self_attn+norms, or full TransformerLayer")
    parser.add_argument("--target_mode", choices=["teacher_path", "student_input"], default="teacher_path",
                        help="teacher_path: target = teacher.layer[0..N](emb). student_input: target = teacher.layer[start..N](student_input_to_start), aligns target to student distribution to mitigate cascading drift.")
    parser.add_argument("--full_layer_from", type=int, default=999999,
                        help="Backward-compatible override: train full TransformerLayer for endpoints >= this index")
    parser.add_argument("--debug_trace", action="store_true",
                        help="Print fine-grained per-step stage traces for hang diagnosis")
    parser.add_argument("--trace_interval", type=int, default=10,
                        help="Step interval for debug_trace logs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--single_process", action="store_true",
                        help="Run without paddle.distributed.launch/env communication. Use CUDA_VISIBLE_DEVICES to choose one local GPU.")
    args = parser.parse_args()

    import paddle
    import paddle.distributed as dist
    import paddle.distributed.fleet as fleet
    import paddlefleet
    import paddlefleet.tensor_parallel.random as _rng_module
    import paddlefleet.parallel_state as _ps

    # Initialize distributed only when launched by paddle.distributed.launch.
    # For local smoke/refine runs, --single_process avoids cluster/env communication hangs.
    if args.single_process:
        rank = 0
        world_size = 1
    else:
        dist.init_parallel_env()
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # gpt_builder only reads this field; no need to call fleet.init/hybrid communication.
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": world_size,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
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

    # Monkey-patch ColumnParallelLinear/RowParallelLinear for TP=1
    from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    def _simple_linear(input_, weight, bias=None):
        # paddle.matmul on [B, S, H] x [H, O] can hit CUBLAS failures on B30Z
        # for long sequence bf16 tensors. Flatten to 2D to use the stable GEMM path.
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

    def trace(msg, step=None, force=False):
        if not args.debug_trace:
            return
        if step is not None and not force and step % args.trace_interval != 0:
            return
        print(f"[TRACE rank={rank}] {msg}", flush=True)

    # =========================================================================
    # Step 1: Load GQA model
    # =========================================================================
    log("=" * 60)
    log("Step 1: Loading GQA model and collecting embedding outputs")
    log(f"  world_size={world_size}, rank={rank}")
    log("=" * 60)

    provider = create_provider(args.gqa_model_config)
    provider.seq_length = args.seq_length
    provider.max_sequence_length = args.seq_length
    gqa_model = provider.provide()

    from safetensors import safe_open
    gqa_param_dict = dict(gqa_model.named_parameters())
    sf_files = sorted([f for f in os.listdir(args.gqa_checkpoint) if f.endswith('.safetensors')])
    gqa_loaded = 0
    for sf in sf_files:
        with safe_open(os.path.join(args.gqa_checkpoint, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                val = f.get_tensor(key).float().numpy()
                pname = ckpt_key_to_pipeline_name(key)
                param = gqa_param_dict.get(pname) if pname else None
                if param is None:
                    for candidate in [key, "model." + key]:
                        param = gqa_param_dict.get(candidate)
                        if param is not None:
                            break
                if param is not None and list(param.shape) == list(val.shape):
                    param.set_value(paddle.to_tensor(val).cast(param.dtype))
                    gqa_loaded += 1
    log(f"  Loaded {gqa_loaded}/{len(gqa_param_dict)} GQA params")

    gqa_model.eval()
    for p in gqa_model.parameters():
        p.stop_gradient = True

    tl_indices = [i for i, fn in enumerate(gqa_model.run_function)
                  if isinstance(fn, TransformerLayer)]
    num_layers = len(tl_indices)
    log(f"  Found {num_layers} TransformerLayers")

    # Load token data - all ranks load, then shard
    if args.data_path:
        from paddleformers.data.indexed_dataset import MMapIndexedDataset
        mmap_ds = MMapIndexedDataset(args.data_path, skip_warmup=True)
        log(f"  Loaded mmap dataset: {len(mmap_ds)} samples available")
        token_ids = []
        idx = 0
        while len(token_ids) < args.num_samples and idx < len(mmap_ds):
            toks = mmap_ds[idx]
            if len(toks) >= args.seq_length:
                token_ids.append(toks[:args.seq_length].astype(np.int64))
            idx += 1
        token_ids = np.stack(token_ids, axis=0)
        log(f"  Collected {token_ids.shape[0]} real samples (seq_length={args.seq_length})")
    else:
        token_ids = np.random.randint(100, 151000, (args.num_samples, args.seq_length))
        log(f"  Using random tokens ({args.num_samples} samples)")

    # Shard data across ranks
    N_total = token_ids.shape[0]
    shard_size = N_total // world_size
    shard_start = rank * shard_size
    shard_end = shard_start + shard_size if rank < world_size - 1 else N_total
    token_ids = token_ids[shard_start:shard_end]
    N_local = token_ids.shape[0]
    log(f"  Total {N_total} samples, ~{shard_size} per GPU")

    # Collect embedding outputs for local shard
    all_emb_outputs = []
    with paddle.no_grad():
        for batch_start in range(0, N_local, args.batch_size):
            batch_end = min(batch_start + args.batch_size, N_local)
            batch = token_ids[batch_start:batch_end]
            input_ids = paddle.to_tensor(batch.astype(np.int64))
            x = gqa_model.run_function[0]({"input_ids": input_ids})
            all_emb_outputs.append(x["hidden_states"].detach().cast("float32").numpy())

    gqa_emb_output = np.concatenate(all_emb_outputs, axis=0)
    del all_emb_outputs
    log(f"  Embedding output per rank: {gqa_emb_output.shape}")

    # Save rotary_pos_emb
    with paddle.no_grad():
        dummy_input = paddle.to_tensor(np.zeros((1, args.seq_length), dtype=np.int64))
        emb_out = gqa_model.run_function[0]({"input_ids": dummy_input})
        rotary_pos_emb_np = emb_out["rotary_pos_emb"].detach().cast("float32").numpy()

    # =========================================================================
    # Step 2: Load VHA model
    # =========================================================================
    log("\n" + "=" * 60)
    log("Step 2: Loading VHA model for cascading refinement (bfloat16)")
    log("=" * 60)

    vha_provider = create_provider(args.vha_model_config)
    vha_provider.seq_length = args.seq_length
    vha_provider.max_sequence_length = args.seq_length
    vha_model = vha_provider.provide()

    param_dict = dict(vha_model.named_parameters())
    vha_sf_files = sorted([f for f in os.listdir(args.vha_checkpoint) if f.endswith('.safetensors')])
    vha_loaded = 0
    vha_shape_mismatch = []
    import ml_dtypes
    for sf in vha_sf_files:
        with safe_open(os.path.join(args.vha_checkpoint, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                val = f.get_tensor(key).float().numpy()
                pname = ckpt_key_to_pipeline_name(key)
                param = param_dict.get(pname) if pname else None
                if param is None:
                    for candidate in [key, key.replace("model.", "")]:
                        param = param_dict.get(candidate)
                        if param is not None:
                            break
                if param is not None:
                    if list(param.shape) == list(val.shape):
                        param.set_value(paddle.to_tensor(val).cast(param.dtype))
                        vha_loaded += 1
                    else:
                        vha_shape_mismatch.append((key, list(val.shape), list(param.shape)))
    log(f"  Loaded {vha_loaded}/{len(param_dict)} VHA params")
    if vha_shape_mismatch:
        log(f"  [WARN] {len(vha_shape_mismatch)} VHA params shape-mismatch (skipped):")
        for k, vs, ps in vha_shape_mismatch[:10]:
            log(f"    {k}: ckpt={vs} model={ps}")
        raise RuntimeError(
            f"Refusing to continue: {len(vha_shape_mismatch)} VHA params silently skipped due to shape mismatch. "
            f"Likely vha_model_config does not match vha_checkpoint architecture.")

    # Keep model parameters in the dtype defined by model config.

    vha_model.eval()

    vha_tl_indices = [i for i, fn in enumerate(vha_model.run_function)
                      if isinstance(fn, TransformerLayer)]
    log(f"  Found {len(vha_tl_indices)} VHA TransformerLayers")

    # =========================================================================
    # Cascading refinement loop (distributed)
    # Each rank optimizes with its own data shard, gradients are all-reduced
    # =========================================================================
    import math

    def get_layer_lr(base_lr, layer_idx, decay_rate):
        """Compute per-layer lr with exponential decay."""
        return base_lr * (decay_rate ** layer_idx)

    def get_cosine_lr(step, total_steps, warmup_steps, max_lr, min_lr):
        """Cosine schedule with linear warmup."""
        if step < warmup_steps:
            return min_lr + (max_lr - min_lr) * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    log(f"\n  Cascading refinement: {num_layers} layers, {N_total} total samples "
        f"({N_local}/rank), {args.refine_steps} steps/endpoint, base_lr={args.lr}, "
        f"decay_rate={args.lr_decay_rate}, patience={args.patience}, "
        f"train_window={args.train_window}, segment_stride={args.segment_stride}, "
        f"train_mode={args.train_mode}")

    if args.train_window < 1:
        raise ValueError("--train_window must be >= 1")
    if args.segment_stride < 1:
        raise ValueError("--segment_stride must be >= 1")

    first_endpoint = min(args.train_window - 1, num_layers - 1)
    endpoint_layers = list(range(first_endpoint, num_layers, args.segment_stride))
    if not endpoint_layers or endpoint_layers[-1] != num_layers - 1:
        endpoint_layers.append(num_layers - 1)

    for layer_idx in endpoint_layers:
        # Defensive barrier: ensure all ranks enter each segment together.
        if world_size > 1:
            dist.barrier()
        train_start_layer = max(0, layer_idx - args.train_window + 1)
        train_end_layer = layer_idx
        layer_lr = get_layer_lr(args.lr, train_start_layer, args.lr_decay_rate)
        log(f"\n{'='*40}")
        log(f"--- Cascading Refine Segment {train_start_layer}..{train_end_layer} "
            f"(target layer {layer_idx}, lr={layer_lr:.6f}) ---")
        log(f"{'='*40}")

        # --- Compute VHA input for the train window (local shard) ---
        vha_inputs_for_layer = []
        with paddle.no_grad():
            for batch_start in range(0, N_local, args.batch_size):
                batch_end = min(batch_start + args.batch_size, N_local)
                h = paddle.to_tensor(gqa_emb_output[batch_start:batch_end]).cast("bfloat16")
                rpe = paddle.to_tensor(rotary_pos_emb_np).cast("bfloat16")
                x_dict = {"hidden_states": h, "rotary_pos_emb": rpe}
                for prev_l in range(train_start_layer):
                    vha_layer_fn = vha_model.run_function[vha_tl_indices[prev_l]]
                    x_dict = vha_layer_fn(x_dict)
                vha_inputs_for_layer.append(
                    x_dict["hidden_states"].detach().cast("float32").numpy()
                )

        vha_input_np = np.concatenate(vha_inputs_for_layer, axis=0)
        del vha_inputs_for_layer
        paddle.device.cuda.empty_cache()
        log(f"  VHA window input shape (per rank): {vha_input_np.shape}")

        # --- Collect target ---
        # teacher_path: target = teacher.layer[0..layer_idx](embedding)
        # student_input: target = teacher.layer[train_start..layer_idx](vha_input)
        gqa_layer_target_parts = []
        with paddle.no_grad():
            if args.target_mode == "student_input":
                for batch_start in range(0, N_local, args.batch_size):
                    batch_end = min(batch_start + args.batch_size, N_local)
                    h = paddle.to_tensor(vha_input_np[batch_start:batch_end]).cast("bfloat16")
                    rpe = paddle.to_tensor(rotary_pos_emb_np).cast("bfloat16")
                    x = {"hidden_states": h, "rotary_pos_emb": rpe}
                    for tl in range(train_start_layer, layer_idx + 1):
                        x = gqa_model.run_function[tl_indices[tl]](x)
                    gqa_layer_target_parts.append(
                        x["hidden_states"].detach().cast("float32").numpy()
                    )
            else:
                for batch_start in range(0, N_local, args.batch_size):
                    batch_end = min(batch_start + args.batch_size, N_local)
                    batch = token_ids[batch_start:batch_end]
                    input_ids = paddle.to_tensor(batch.astype(np.int64))
                    x = gqa_model.run_function[0]({"input_ids": input_ids})
                    layer_count = 0
                    for fi in range(1, len(gqa_model.run_function)):
                        fn = gqa_model.run_function[fi]
                        if isinstance(fn, TransformerLayer):
                            x = fn(x)
                            layer_count += 1
                            if layer_count > layer_idx:
                                break
                        else:
                            x = fn(x)
                    gqa_layer_target_parts.append(
                        x["hidden_states"].detach().cast("float32").numpy()
                    )
        Y_np = np.concatenate(gqa_layer_target_parts, axis=0)
        del gqa_layer_target_parts
        paddle.device.cuda.empty_cache()
        log(f"  target_mode={args.target_mode}, target shape: {Y_np.shape}")

        # --- Optimize selected VHA layers ---

        train_params = []
        train_layer_set = set(range(train_start_layer, train_end_layer + 1))
        effective_train_mode = "full" if layer_idx >= args.full_layer_from else args.train_mode
        for layer_i, tl_idx in enumerate(vha_tl_indices):
            layer_fn = vha_model.run_function[tl_idx]
            layer_in_window = layer_i in train_layer_set
            for pname, param in layer_fn.named_parameters():
                if effective_train_mode == "full":
                    should_train = layer_in_window
                elif effective_train_mode == "attn_norm":
                    should_train = layer_in_window and (
                        "self_attn" in pname or "norm" in pname or "layernorm" in pname
                    )
                else:
                    should_train = layer_in_window and "self_attn" in pname
                param.stop_gradient = not should_train
                if should_train:
                    train_params.append(param)

        if not train_params:
            log(f"  No trainable params, skipping")
            continue

        train_desc = f"{effective_train_mode} params in layers {train_start_layer}..{train_end_layer}"
        log(f"  Optimizing {len(train_params)} params in {train_desc} "
            f"({sum(p.numel().item() for p in train_params)} elements)")

        warmup_steps = int(args.refine_steps * args.warmup_ratio)
        min_lr = layer_lr * args.lr_min_ratio
        # Per-layer Adam beta1: reduce momentum for later layers to avoid oscillation
        layer_beta1 = min(0.9, 0.9 * (args.adam_beta1_decay ** layer_idx))
        optimizer = paddle.optimizer.Adam(
            learning_rate=layer_lr,
            beta1=layer_beta1,
            beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
            parameters=train_params,
            multi_precision=True
        )
        log(f"  layer_lr={layer_lr:.6f}, beta1={layer_beta1:.4f}, "
            f"beta2={args.adam_beta2}, wd={args.weight_decay}, "
            f"grad_accum={args.grad_accum}")

        mini_batch = min(args.batch_size, N_local)
        initial_loss = None
        best_loss = float('inf')
        best_param_states = None  # Save best checkpoint for rollback
        steps_without_improvement = 0
        actual_steps = 0

        for step in range(args.refine_steps):
            trace(f"L{train_start_layer}-{layer_idx} S{step} begin", step)
            # Cosine lr schedule with warmup
            current_lr = get_cosine_lr(step, args.refine_steps, warmup_steps, layer_lr, min_lr)
            optimizer.set_lr(current_lr)

            # Gradient accumulation: accumulate over multiple micro-batches
            accum_loss = 0.0
            for _accum_i in range(args.grad_accum):
                trace(f"L{train_start_layer}-{layer_idx} S{step} A{_accum_i} sample", step)
                indices = np.random.choice(N_local, mini_batch, replace=False)
                X_batch = paddle.to_tensor(vha_input_np[indices]).cast("bfloat16")
                X_batch.stop_gradient = True
                Y_batch = paddle.to_tensor(Y_np[indices])

                rpe = paddle.to_tensor(rotary_pos_emb_np).cast("bfloat16")
                rpe.stop_gradient = True
                x_dict = {"hidden_states": X_batch, "rotary_pos_emb": rpe}
                trace(f"L{train_start_layer}-{layer_idx} S{step} A{_accum_i} forward_start", step)
                for train_l in range(train_start_layer, layer_idx + 1):
                    layer_fn = vha_model.run_function[vha_tl_indices[train_l]]
                    x_dict = layer_fn(x_dict)
                trace(f"L{train_start_layer}-{layer_idx} S{step} A{_accum_i} forward_done", step)
                pred = x_dict["hidden_states"]

                loss = paddle.nn.functional.mse_loss(pred.cast("float32"), Y_batch)
                # Scale loss by accum steps so gradients are averaged
                scaled_loss = loss / args.grad_accum
                trace(f"L{train_start_layer}-{layer_idx} S{step} A{_accum_i} backward_start", step)
                scaled_loss.backward()
                trace(f"L{train_start_layer}-{layer_idx} S{step} A{_accum_i} backward_done", step)
                accum_loss += loss.item()
                trace(f"L{train_start_layer}-{layer_idx} S{step} A{_accum_i} loss_item_done", step)

            accum_loss /= args.grad_accum

            # All-reduce loss across ranks so all ranks make identical
            # early-stop decisions. Without this, ranks can desync (some
            # exiting the loop earlier than others) and deadlock at the
            # next collective op in subsequent segments.
            if world_size > 1:
                _loss_t = paddle.to_tensor([accum_loss], dtype="float32")
                dist.all_reduce(_loss_t, op=dist.ReduceOp.SUM)
                accum_loss = float(_loss_t.item()) / world_size

            # All-reduce gradients across ranks
            trace(f"L{train_start_layer}-{layer_idx} S{step} allreduce_start", step)
            for param_idx, param in enumerate(train_params):
                if param.grad is not None:
                    if world_size > 1:
                        trace(f"L{train_start_layer}-{layer_idx} S{step} allreduce_param_{param_idx}_start", step)
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        trace(f"L{train_start_layer}-{layer_idx} S{step} allreduce_param_{param_idx}_done", step)
                        param.grad = param.grad * (1.0 / world_size)
            trace(f"L{train_start_layer}-{layer_idx} S{step} allreduce_done", step)

            # Gradient clipping in fp32 to avoid mixed bf16/fp32 stack errors
            if args.grad_clip > 0:
                trace(f"L{train_start_layer}-{layer_idx} S{step} clip_start", step)
                grad_norm_sq = None
                for param in train_params:
                    if param.grad is not None:
                        grad_sq = paddle.sum(paddle.square(param.grad.cast("float32")))
                        grad_norm_sq = grad_sq if grad_norm_sq is None else grad_norm_sq + grad_sq
                if grad_norm_sq is not None:
                    grad_norm = paddle.sqrt(grad_norm_sq)
                    clip_coef = args.grad_clip / (grad_norm + 1e-6)
                    if clip_coef.item() < 1.0:
                        for param in train_params:
                            if param.grad is not None:
                                param.grad = param.grad * clip_coef.cast(param.grad.dtype)

            trace(f"L{train_start_layer}-{layer_idx} S{step} clip_done", step)
            trace(f"L{train_start_layer}-{layer_idx} S{step} optim_step_start", step)
            optimizer.step()
            trace(f"L{train_start_layer}-{layer_idx} S{step} optim_step_done", step)
            optimizer.clear_grad()
            trace(f"L{train_start_layer}-{layer_idx} S{step} clear_grad_done", step)

            loss_val = accum_loss
            if initial_loss is None:
                initial_loss = loss_val
            actual_steps = step + 1

            # Best checkpoint tracking and early stopping
            if loss_val < best_loss - 1e-8:
                best_loss = loss_val
                steps_without_improvement = 0
                # Save best param state (trainable params only)
                best_param_states = [p.numpy().copy() for p in train_params]
            else:
                steps_without_improvement += 1

            best_impr = (1 - best_loss / initial_loss) if initial_loss and initial_loss > 0 else 0
            if rank == 0 and (step % 100 == 0 or step == args.refine_steps - 1):
                impr = (1 - loss_val / initial_loss) * 100 if initial_loss > 0 else 0
                print(f"    Step {step}: MSE={loss_val:.6f} ({impr:.1f}% reduction) "
                      f"best={best_loss:.6f} ({best_impr * 100:.1f}% best) "
                      f"lr={current_lr:.6f} no_impr={steps_without_improvement}",
                      flush=True)

            should_stop_target = step + 1 >= args.min_steps and best_impr >= args.target_reduction
            if should_stop_target or steps_without_improvement >= args.patience:
                if rank == 0:
                    reason = "target_reduction" if should_stop_target else "patience"
                    print(f"    Early stop at step {step} ({reason}): best_MSE={best_loss:.6f} "
                          f"({best_impr * 100:.1f}% reduction)", flush=True)
                break

        # Rollback to best checkpoint
        if best_param_states is not None:
            for param, best_val in zip(train_params, best_param_states):
                param.set_value(paddle.to_tensor(best_val).cast(param.dtype))
            log(f"  Rolled back to best checkpoint (MSE={best_loss:.6f})")

        log(f"  Segment {train_start_layer}..{layer_idx} done: {actual_steps} steps, "
            f"loss {initial_loss:.6f} -> {best_loss:.6f}")

        # Freeze params after refinement
        for param in train_params:
            param.stop_gradient = True
        del optimizer, vha_input_np, Y_np, best_param_states
        gc.collect()
        paddle.device.cuda.empty_cache()

        # Sync params: broadcast from rank 0 to ensure consistency
        if world_size > 1:
            for param in train_params:
                dist.broadcast(param, src=0)

    # =========================================================================
    # Step 3: Save refined VHA checkpoint (rank 0 only)
    # =========================================================================
    if rank == 0:
        print("\n" + "=" * 60, flush=True)
        print("Step 3: Saving cascading-refined VHA checkpoint", flush=True)
        print("=" * 60, flush=True)

        os.makedirs(args.output_path, exist_ok=True)
        from safetensors.numpy import save_file

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

        # Handle tie_word_embeddings: pipeline model may have no separate lm_head.
        if (
            "model.lm_head.weight" not in output_tensors
            and "model.embedding.embed_tokens.weight" in output_tensors
        ):
            output_tensors["model.lm_head.weight"] = output_tensors[
                "model.embedding.embed_tokens.weight"
            ]

        output_file = os.path.join(args.output_path, "model-00001-of-00001.safetensors")
        print(f"  Saving {len(output_tensors)} tensors to {output_file}", flush=True)
        save_file(output_tensors, output_file)

        weight_map = {k: "model-00001-of-00001.safetensors" for k in output_tensors.keys()}
        total_size = sum(t.nbytes for t in output_tensors.values())
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(args.output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=4)

        src_config = os.path.join(args.vha_checkpoint, "config.json")
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(args.output_path, "config.json"))

        print(f"\nDone! Cascading-refined checkpoint saved to {args.output_path}", flush=True)

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
