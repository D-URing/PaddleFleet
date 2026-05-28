#!/usr/bin/env python3
"""Per-layer gradient refinement for VHA checkpoint.

Key fix: model must stay in bfloat16 so that flash attention path
(with is_causal=True) is used. Float32 falls back to manual attention
which does NOT apply causal masking (dot_product_attention.py line 406).
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
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_checkpoint", type=str, required=True)
    parser.add_argument("--gqa_model_config", type=str, required=True)
    parser.add_argument("--vha_checkpoint", type=str, required=True)
    parser.add_argument("--vha_model_config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--refine_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    import paddle
    import paddlefleet
    import paddlefleet.tensor_parallel.random as _rng_module
    import paddlefleet.parallel_state as _ps

    _rng_module.initialize_rng_tracker()
    _rng_module._CUDA_RNG_STATE_TRACKER.fork = lambda name="model-parallel-rng": contextlib.nullcontext()
    _ps.get_tensor_model_parallel_rank = lambda: 0
    _ps.get_tensor_model_parallel_world_size = lambda: 1
    _ps.get_pipeline_model_parallel_rank = lambda: 0
    _ps.get_pipeline_model_parallel_world_size = lambda: 1
    _ps.get_data_parallel_rank = lambda: 0
    _ps.get_data_parallel_world_size = lambda: 1
    _ps.get_expert_model_parallel_rank = lambda: 0
    _ps.get_expert_tensor_parallel_rank = lambda: 0
    _ps.get_expert_tensor_and_model_parallel_rank = lambda: 0

    # Monkey-patch ColumnParallelLinear/RowParallelLinear to use simple matmul.
    # Mathematically equivalent for TP=1, avoids PyLayer backward issues.
    # IMPORTANT: must handle skip_bias_add correctly to match real behavior.
    from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    def _simple_col_fwd(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else self.weight
        out = paddle.matmul(input_, w)
        if not self.skip_bias_add and self.bias is not None:
            out = out + self.bias
        output_bias = self.bias if self.skip_bias_add and self.bias is not None else None
        return out, output_bias

    def _simple_row_fwd(self, input_, weight=None):
        w = weight if weight is not None else self.weight
        out = paddle.matmul(input_, w)
        if not self.skip_bias_add and self.bias is not None:
            out = out + self.bias
        output_bias = self.bias if self.skip_bias_add and self.bias is not None else None
        return out, output_bias

    ColumnParallelLinear.forward = _simple_col_fwd
    RowParallelLinear.forward = _simple_row_fwd

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'VHA'))
    from models.qwen_provider import create_provider
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    # =========================================================================
    # Step 1: Load GQA model, collect per-layer hidden states (in bfloat16)
    # =========================================================================
    print("=" * 60, flush=True)
    print("Step 1: Collecting GQA per-layer hidden states", flush=True)
    print("=" * 60, flush=True)

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
    print(f"  Loaded {gqa_loaded}/{len(gqa_param_dict)} GQA params", flush=True)

    gqa_model.eval()
    for p in gqa_model.parameters():
        p.stop_gradient = True

    tl_indices = [i for i, fn in enumerate(gqa_model.run_function)
                  if isinstance(fn, TransformerLayer)]
    num_layers = len(tl_indices)
    print(f"  Found {num_layers} TransformerLayers", flush=True)

    token_ids = np.random.randint(100, 151000, (args.num_samples, args.seq_length))
    print(f"  Running manual forward ({args.num_samples} samples)...", flush=True)

    all_layer_inputs = [[] for _ in range(num_layers)]
    all_layer_outputs = [[] for _ in range(num_layers)]

    with paddle.no_grad():
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_samples)
            batch = token_ids[batch_start:batch_end]
            input_ids = paddle.to_tensor(batch.astype(np.int64))
            x = gqa_model.run_function[0]({"input_ids": input_ids})
            layer_count = 0
            for i in range(1, len(gqa_model.run_function)):
                fn = gqa_model.run_function[i]
                if isinstance(fn, TransformerLayer):
                    h_in = x["hidden_states"].detach().cast("float32").numpy()
                    all_layer_inputs[layer_count].append(h_in)
                    x = fn(x)
                    h_out = x["hidden_states"].detach().cast("float32").numpy()
                    all_layer_outputs[layer_count].append(h_out)
                    layer_count += 1
                    if layer_count >= num_layers:
                        break
                else:
                    x = fn(x)
            if batch_start % (args.batch_size * 4) == 0:
                print(f"    Batch {batch_start // args.batch_size}: done", flush=True)

    gqa_inputs = [np.concatenate(all_layer_inputs[i], axis=0) for i in range(num_layers)]
    gqa_outputs = [np.concatenate(all_layer_outputs[i], axis=0) for i in range(num_layers)]
    del all_layer_inputs, all_layer_outputs
    print(f"  Layer 0: input={gqa_inputs[0].shape}, output={gqa_outputs[0].shape}", flush=True)
    N_total = gqa_inputs[0].shape[0]

    # Save rotary_pos_emb from embedding layer
    with paddle.no_grad():
        dummy_input = paddle.to_tensor(np.zeros((1, args.seq_length), dtype=np.int64))
        emb_out = gqa_model.run_function[0]({"input_ids": dummy_input})
        rotary_pos_emb_np = emb_out["rotary_pos_emb"].detach().cast("float32").numpy()
    print(f"  rotary_pos_emb shape: {rotary_pos_emb_np.shape}", flush=True)

    del gqa_model
    paddle.device.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Step 2: Load VHA model and refine per-layer
    # CRITICAL: Keep model in bfloat16 so flash attention (causal) is used!
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("Step 2: Loading VHA model and refining per-layer (bfloat16)", flush=True)
    print("=" * 60, flush=True)

    vha_provider = create_provider(args.vha_model_config)
    vha_provider.seq_length = args.seq_length
    vha_provider.max_sequence_length = args.seq_length
    vha_model = vha_provider.provide()

    param_dict = dict(vha_model.named_parameters())
    vha_sf_files = sorted([f for f in os.listdir(args.vha_checkpoint) if f.endswith('.safetensors')])
    vha_loaded = 0
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
                if param is not None and list(param.shape) == list(val.shape):
                    # Load as bfloat16 to match model dtype
                    import ml_dtypes
                    bf16_arr = val.astype(ml_dtypes.bfloat16)
                    param.value().get_tensor().set(bf16_arr, paddle.CUDAPlace(0))
                    vha_loaded += 1
    print(f"  Loaded {vha_loaded}/{len(param_dict)} VHA params", flush=True)

    # Ensure ALL params are bfloat16 (premix/postmix may be created as float32)
    cast_count = 0
    for name, param in vha_model.named_parameters():
        if param.dtype != paddle.bfloat16:
            import ml_dtypes
            arr = param.cast("float32").numpy().astype(ml_dtypes.bfloat16)
            param.value().get_tensor().set(arr, paddle.CUDAPlace(0))
            cast_count += 1
    if cast_count > 0:
        print(f"  Cast {cast_count} params to bfloat16", flush=True)

    vha_model.eval()

    vha_tl_indices = [i for i, fn in enumerate(vha_model.run_function)
                      if isinstance(fn, TransformerLayer)]
    print(f"  Found {len(vha_tl_indices)} VHA TransformerLayers", flush=True)

    print(f"\n  Refining {num_layers} layers, {N_total} samples, {args.refine_steps} steps, lr={args.lr}", flush=True)

    for layer_idx in range(num_layers):
        print(f"\n--- Refining Layer {layer_idx} ---", flush=True)
        vha_layer = vha_model.run_function[vha_tl_indices[layer_idx]]

        attn_params = []
        for pname, param in vha_layer.named_parameters():
            if 'self_attn' in pname:
                param.stop_gradient = False
                attn_params.append(param)
            else:
                param.stop_gradient = True

        if not attn_params:
            print(f"  No attention params, skipping", flush=True)
            continue

        print(f"  Optimizing {len(attn_params)} attn params ({sum(p.numel().item() for p in attn_params)} elements)", flush=True)

        X_np = gqa_inputs[layer_idx]
        Y_np = gqa_outputs[layer_idx]

        optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=attn_params)
        mini_batch = min(args.batch_size, N_total)
        initial_loss = None

        for step in range(args.refine_steps):
            indices = np.random.choice(N_total, mini_batch, replace=False)
            # CRITICAL: input must be bfloat16 so flash attention path is taken
            X_batch = paddle.to_tensor(X_np[indices]).cast("bfloat16")
            X_batch.stop_gradient = False
            Y_batch = paddle.to_tensor(Y_np[indices])  # float32 for loss computation

            rpe = paddle.to_tensor(rotary_pos_emb_np).cast("bfloat16")
            rpe.stop_gradient = False
            layer_input = {"hidden_states": X_batch, "rotary_pos_emb": rpe}
            out = vha_layer(layer_input)
            pred = out["hidden_states"]

            # Loss in float32 for numerical stability
            loss = paddle.nn.functional.mse_loss(pred.cast("float32"), Y_batch)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            loss_val = loss.item()
            if initial_loss is None:
                initial_loss = loss_val
            if step % 50 == 0 or step == args.refine_steps - 1:
                impr = (1 - loss_val / initial_loss) * 100 if initial_loss > 0 else 0
                print(f"    Step {step}: MSE={loss_val:.6f} ({impr:.1f}% reduction)", flush=True)

        for param in attn_params:
            param.stop_gradient = True
        del X_np, Y_np
        gqa_inputs[layer_idx] = None
        gqa_outputs[layer_idx] = None
        gc.collect()
        paddle.device.cuda.empty_cache()

    # =========================================================================
    # Step 3: Save refined VHA checkpoint
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("Step 3: Saving refined VHA checkpoint", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(args.output_path, exist_ok=True)
    import ml_dtypes
    from safetensors.numpy import save_file

    output_tensors = {}
    for name, param in vha_model.named_parameters():
        val = param.cast("float32").numpy().astype(ml_dtypes.bfloat16)
        parts = name.split(".", 1)
        idx = int(parts[0])
        rest = parts[1]
        if idx == 0:
            out_key = f"model.{rest}"
        else:
            out_key = f"model.layers.{idx - 1}.{rest}"
        output_tensors[out_key] = val

    # Fix key mapping for final norm and lm_head
    if "model.layers.28.norm.weight" in output_tensors and "model.norm.weight" not in output_tensors:
        output_tensors["model.norm.weight"] = output_tensors.pop("model.layers.28.norm.weight")
    if "model.layers.29.weight" in output_tensors:
        output_tensors["model.lm_head.weight"] = output_tensors.pop("model.layers.29.weight")
    # Ensure lm_head exists (tied weights — embedding may be embed_tokens or word_embeddings)
    if "model.lm_head.weight" not in output_tensors:
        for emb_key in ["model.embedding.embed_tokens.weight", "model.embedding.word_embeddings.weight"]:
            if emb_key in output_tensors:
                output_tensors["model.lm_head.weight"] = output_tensors[emb_key].copy()
                break

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

    print(f"\nDone! Refined checkpoint saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
