#!/usr/bin/env python3
"""Evaluate per-layer attention-output MSE between GQA teacher and VHA checkpoint."""

import argparse
import contextlib
import json
import os
import sys

import numpy as np
from safetensors import safe_open


def ckpt_key_to_pipeline_name(key, num_layers=28):
    k = key.replace("model.", "", 1) if key.startswith("model.") else key
    if k.startswith("embedding."):
        return f"0.{k}"
    if k.startswith("layers."):
        parts = k.split(".", 2)
        layer_idx = int(parts[1])
        return f"{layer_idx + 1}.{parts[2]}"
    if k == "norm.weight":
        return f"{num_layers + 1}.norm.weight"
    if k == "lm_head.weight":
        return f"{num_layers + 2}.weight"
    return None


def setup_paddlefleet_single_gpu():
    import paddlefleet.tensor_parallel.random as rng_module
    import paddlefleet.parallel_state as ps

    rng_module.initialize_rng_tracker()
    rng_module._CUDA_RNG_STATE_TRACKER.fork = lambda name="model-parallel-rng": contextlib.nullcontext()
    ps.get_tensor_model_parallel_rank = lambda: 0
    ps.get_tensor_model_parallel_world_size = lambda: 1
    ps.get_pipeline_model_parallel_rank = lambda: 0
    ps.get_pipeline_model_parallel_world_size = lambda: 1
    ps.get_data_parallel_rank = lambda: 0
    ps.get_data_parallel_world_size = lambda: 1
    ps.get_expert_model_parallel_rank = lambda: 0
    ps.get_expert_tensor_parallel_rank = lambda: 0
    ps.get_expert_tensor_and_model_parallel_rank = lambda: 0

    from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    def simple_linear(input_, weight, bias=None):
        out_shape = list(input_.shape[:-1]) + [weight.shape[-1]]
        flat_input = input_.reshape([-1, input_.shape[-1]]).cast("float32")
        out = flat_input.matmul(weight.cast("float32")).cast(input_.dtype).reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out

    def simple_col_fwd(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else self.weight
        bias = self.bias if not self.skip_bias_add and self.bias is not None else None
        out = simple_linear(input_, w, bias)
        output_bias = self.bias if self.skip_bias_add and self.bias is not None else None
        return out, output_bias

    def simple_row_fwd(self, input_, weight=None):
        w = weight if weight is not None else self.weight
        bias = self.bias if not self.skip_bias_add and self.bias is not None else None
        out = simple_linear(input_, w, bias)
        output_bias = self.bias if self.skip_bias_add and self.bias is not None else None
        return out, output_bias

    ColumnParallelLinear.forward = simple_col_fwd
    RowParallelLinear.forward = simple_row_fwd


def load_safetensors_into_pipeline_model(model, checkpoint_dir, num_layers=28):
    import paddle

    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())
    sf_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")])
    loaded = 0
    for sf in sf_files:
        with safe_open(os.path.join(checkpoint_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                val = f.get_tensor(key).float().numpy()
                candidates = [key, "model." + key, key.replace("model.", "", 1)]
                pname = ckpt_key_to_pipeline_name(key, num_layers)
                if pname is not None:
                    candidates.insert(0, pname)
                target = None
                for candidate in candidates:
                    target = param_dict.get(candidate) or buffer_dict.get(candidate)
                    if target is not None:
                        break
                if target is not None and list(target.shape) == list(val.shape):
                    target.set_value(paddle.to_tensor(val).cast(target.dtype))
                    loaded += 1
    return loaded, len(param_dict)


def load_tokens(data_path, num_samples, seq_length):
    if data_path:
        from paddleformers.data.indexed_dataset import MMapIndexedDataset
        ds = MMapIndexedDataset(data_path, skip_warmup=True)
        token_ids = []
        idx = 0
        while len(token_ids) < num_samples and idx < len(ds):
            toks = ds[idx]
            if len(toks) >= seq_length:
                token_ids.append(toks[:seq_length].astype(np.int64))
            idx += 1
        if token_ids:
            return np.stack(token_ids, axis=0)
    return np.random.randint(0, 151936, (num_samples, seq_length), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_checkpoint", required=True)
    parser.add_argument("--vha_checkpoint", required=True)
    parser.add_argument("--gqa_model_config", required=True)
    parser.add_argument("--vha_model_config", required=True)
    parser.add_argument("--data_path", default=None, help="mmap basename without .bin/.idx")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    import paddle
    setup_paddlefleet_single_gpu()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "VHA"))
    from models.qwen_provider import create_provider
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    def build_model(config_path, checkpoint_path, label):
        provider = create_provider(config_path)
        provider.seq_length = args.seq_length
        provider.max_sequence_length = args.seq_length
        model = provider.provide()
        loaded, total = load_safetensors_into_pipeline_model(model, checkpoint_path)
        print(f"Loaded {loaded}/{total} {label} params", flush=True)
        model.eval()
        for p in model.parameters():
            p.stop_gradient = True
        return model

    gqa_model = build_model(args.gqa_model_config, args.gqa_checkpoint, "GQA")
    vha_model = build_model(args.vha_model_config, args.vha_checkpoint, "VHA")
    gqa_layers = [fn for fn in gqa_model.run_function if isinstance(fn, TransformerLayer)]
    vha_layers = [fn for fn in vha_model.run_function if isinstance(fn, TransformerLayer)]
    token_ids = load_tokens(args.data_path, args.num_samples, args.seq_length)

    def call_attn(layer, hidden_states, rotary_pos_emb):
        return layer.self_attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)[0]

    layer_sse = [0.0 for _ in gqa_layers]
    layer_ref_sse = [0.0 for _ in gqa_layers]
    layer_count = [0 for _ in gqa_layers]

    with paddle.no_grad():
        for batch_start in range(0, token_ids.shape[0], args.batch_size):
            batch = token_ids[batch_start:batch_start + args.batch_size]
            input_ids = paddle.to_tensor(batch.astype(np.int64))
            gqa_x = gqa_model.run_function[0]({"input_ids": input_ids})
            vha_x = vha_model.run_function[0]({"input_ids": input_ids})
            for layer_idx, (gqa_layer, vha_layer) in enumerate(zip(gqa_layers, vha_layers)):
                rpe = gqa_x.get("rotary_pos_emb")
                gqa_hidden = gqa_x["hidden_states"]
                vha_hidden = vha_x["hidden_states"]
                gqa_out = call_attn(gqa_layer, gqa_hidden, rpe).cast("float32")
                vha_out = call_attn(vha_layer, vha_hidden, rpe).cast("float32")
                diff = vha_out - gqa_out
                layer_sse[layer_idx] += float(paddle.sum(diff * diff).item())
                layer_ref_sse[layer_idx] += float(paddle.sum(gqa_out * gqa_out).item())
                layer_count[layer_idx] += int(np.prod(gqa_out.shape))
                gqa_x = gqa_layer(gqa_x)
                vha_x = vha_layer(vha_x)

    results = []
    for layer_idx in range(len(gqa_layers)):
        mse = layer_sse[layer_idx] / max(layer_count[layer_idx], 1)
        rel = (layer_sse[layer_idx] / max(layer_ref_sse[layer_idx], 1e-12)) ** 0.5
        results.append({"layer": layer_idx, "mse": mse, "relative_rmse": rel})
        print(f"layer {layer_idx:02d}: attn_mse={mse:.6e}, rel_rmse={rel:.6f}", flush=True)

    mean_mse = sum(r["mse"] for r in results) / len(results)
    mean_rel = sum(r["relative_rmse"] for r in results) / len(results)
    summary = {"mean_mse": mean_mse, "mean_relative_rmse": mean_rel, "layers": results}
    print(f"mean: attn_mse={mean_mse:.6e}, rel_rmse={mean_rel:.6f}", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
