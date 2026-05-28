#!/usr/bin/env python3
"""Alignment-first VHA refinement.

This refines a directly converted VHA checkpoint against the GQA teacher with
cascading block-output MSE. It is intentionally conservative: by default only
VHA structural parameters and attention projections are trainable, while MLP,
embedding, final norm, and LM head stay frozen.
"""

import argparse
import contextlib
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


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


def setup_single_gpu_runtime():
    import paddle.distributed.fleet as fleet
    import paddlefleet.parallel_state as ps
    import paddlefleet.tensor_parallel.random as rng_module

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {"dp_degree": 1, "mp_degree": 1, "pp_degree": 1, "sharding_degree": 1}
    fleet.fleet._user_defined_strategy = strategy

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
        import paddle

        out_shape = list(input_.shape[:-1]) + [weight.shape[-1]]
        flat_input = input_.reshape([-1, input_.shape[-1]]).cast("float32")
        out = paddle.matmul(flat_input, weight.cast("float32")).cast(input_.dtype).reshape(out_shape)
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


def load_tokens(data_path, num_samples, seq_length):
    if data_path:
        from paddleformers.data.indexed_dataset import MMapIndexedDataset

        dataset = MMapIndexedDataset(data_path, skip_warmup=True)
        token_ids = []
        idx = 0
        while len(token_ids) < num_samples and idx < len(dataset):
            tokens = dataset[idx]
            if len(tokens) >= seq_length:
                token_ids.append(tokens[:seq_length].astype(np.int64))
            idx += 1
        if token_ids:
            return np.stack(token_ids, axis=0)
    return np.random.randint(100, 151000, (num_samples, seq_length), dtype=np.int64)


def load_safetensors_into_model(model, checkpoint_dir, num_layers=28):
    import paddle

    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())
    sf_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")])
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
    loaded = 0
    skipped = []
    for sf in sf_files:
        with safe_open(os.path.join(checkpoint_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                value = f.get_tensor(key).float().numpy()
                candidates = []
                pipeline_name = ckpt_key_to_pipeline_name(key, num_layers=num_layers)
                if pipeline_name is not None:
                    candidates.append(pipeline_name)
                candidates.extend([key, "model." + key, key.replace("model.", "", 1)])
                target = None
                target_name = None
                for candidate in candidates:
                    if candidate in param_dict:
                        target = param_dict[candidate]
                        target_name = candidate
                        break
                    if candidate in buffer_dict:
                        target = buffer_dict[candidate]
                        target_name = candidate
                        break
                if target is not None and list(target.shape) == list(value.shape):
                    target.set_value(paddle.to_tensor(value, place=target.place).cast(target.dtype))
                    loaded += 1
                else:
                    skipped.append(key if target_name is None else f"{key}->{target_name}: {value.shape} vs {list(target.shape)}")
    return loaded, len(param_dict), skipped


def is_trainable_param(name, train_mode):
    if train_mode == "vha_only":
        return "vha_" in name
    if train_mode == "vha_o":
        return "vha_" in name or "o_proj" in name
    if train_mode == "attn":
        return "self_attn" in name
    if train_mode == "attn_norm":
        return "self_attn" in name or "norm" in name or "layernorm" in name
    return True


def save_pipeline_model(model, output_path, source_config):
    import ml_dtypes

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_tensors = {}
    for name, param in model.named_parameters():
        value = param.cast("float32").numpy().astype(ml_dtypes.bfloat16)
        parts = name.split(".", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            output_tensors[name] = value
            continue
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
        output_tensors[out_key] = value
    if "model.lm_head.weight" not in output_tensors and "model.embedding.embed_tokens.weight" in output_tensors:
        output_tensors["model.lm_head.weight"] = output_tensors["model.embedding.embed_tokens.weight"]
    output_file = output_path / "model-00001-of-00001.safetensors"
    save_file(output_tensors, str(output_file))
    weight_map = {key: output_file.name for key in output_tensors}
    total_size = sum(tensor.nbytes for tensor in output_tensors.values())
    (output_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=4))
    if source_config and os.path.exists(source_config):
        shutil.copy(source_config, output_path / "config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_checkpoint", required=True)
    parser.add_argument("--gqa_model_config", required=True)
    parser.add_argument("--vha_checkpoint", required=True)
    parser.add_argument("--vha_model_config", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path", default=None, help="MMap dataset basename without .bin/.idx")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--refine_steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_window", type=int, default=1)
    parser.add_argument("--segment_stride", type=int, default=1)
    parser.add_argument("--start_endpoint_layer", type=int, default=0)
    parser.add_argument("--max_endpoint_layer", type=int, default=-1)
    parser.add_argument("--train_mode", choices=["vha_only", "vha_o", "attn", "attn_norm", "full"], default="vha_o")
    parser.add_argument("--loss_mode", choices=["mse", "logits_kl"], default="mse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    import paddle
    import paddle.nn.functional as F
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    np.random.seed(args.seed)
    paddle.seed(args.seed)
    if paddle.device.is_compiled_with_cuda():
        paddle.set_device("gpu:0")
    setup_single_gpu_runtime()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "VHA"))
    from models.qwen_provider import create_provider

    def build_model(config_path, checkpoint_path, label):
        provider = create_provider(config_path)
        provider.seq_length = args.seq_length
        provider.max_sequence_length = args.seq_length
        model = provider.provide()
        loaded, total, skipped = load_safetensors_into_model(model, checkpoint_path)
        print(f"Loaded {loaded}/{total} {label} params; skipped={len(skipped)}", flush=True)
        if skipped:
            print(f"  first skipped {label}: {skipped[:5]}", flush=True)
        model.eval()
        for param in model.parameters():
            param.stop_gradient = True
        return model

    print("Loading teacher/student", flush=True)
    gqa_model = build_model(args.gqa_model_config, args.gqa_checkpoint, "GQA")
    vha_model = build_model(args.vha_model_config, args.vha_checkpoint, "VHA")
    gqa_layers = [idx for idx, fn in enumerate(gqa_model.run_function) if isinstance(fn, TransformerLayer)]
    vha_layers = [idx for idx, fn in enumerate(vha_model.run_function) if isinstance(fn, TransformerLayer)]
    num_layers = min(len(gqa_layers), len(vha_layers))
    endpoint_limit = num_layers - 1 if args.max_endpoint_layer < 0 else min(args.max_endpoint_layer, num_layers - 1)
    first_endpoint = max(args.start_endpoint_layer, min(args.train_window - 1, endpoint_limit))
    endpoints = list(range(first_endpoint, endpoint_limit + 1, args.segment_stride))
    if endpoints and endpoints[-1] != endpoint_limit:
        endpoints.append(endpoint_limit)

    token_ids = load_tokens(args.data_path, args.num_samples, args.seq_length)
    print(f"Loaded tokens: {token_ids.shape}", flush=True)
    with paddle.no_grad():
        dummy = paddle.to_tensor(np.zeros((1, args.seq_length), dtype=np.int64))
        rotary_pos_emb = gqa_model.run_function[0]({"input_ids": dummy})["rotary_pos_emb"].detach().cast("float32").numpy()

    def run_layers(model, layer_indices, x_dict, start_layer, end_layer):
        for layer_idx in range(start_layer, end_layer + 1):
            x_dict = model.run_function[layer_indices[layer_idx]](x_dict)
        return x_dict

    if args.loss_mode == "logits_kl":
        print(f"\n=== logits_kl refine: layers {first_endpoint}-{endpoint_limit} mode={args.train_mode} ===", flush=True)
        gqa_logits_list = []
        token_ids_list = []
        with paddle.no_grad():
            for batch_start in range(0, token_ids.shape[0], args.batch_size):
                batch = token_ids[batch_start:batch_start + args.batch_size]
                input_ids = paddle.to_tensor(batch.astype(np.int64))
                gqa_x = gqa_model.run_function[0]({"input_ids": input_ids})
                gqa_x = run_layers(gqa_model, gqa_layers, gqa_x, 0, num_layers - 1)
                for fn_idx in range(gqa_layers[-1] + 1, len(gqa_model.run_function)):
                    gqa_x = gqa_model.run_function[fn_idx](gqa_x)
                logits_out = gqa_x if isinstance(gqa_x, paddle.Tensor) else gqa_x["hidden_states"]
                gqa_logits_list.append(logits_out.detach().cast("float32").numpy())
                token_ids_list.append(batch)
        gqa_logits_np = np.concatenate(gqa_logits_list, axis=0)
        token_ids_all = np.concatenate(token_ids_list, axis=0)
        del gqa_logits_list, token_ids_list
        paddle.device.cuda.empty_cache()
        train_params = []
        train_names = []
        for layer_i, tl_idx in enumerate(vha_layers):
            layer = vha_model.run_function[tl_idx]
            layer_in_range = first_endpoint <= layer_i <= endpoint_limit
            for name, param in layer.named_parameters():
                should_train = layer_in_range and is_trainable_param(name, args.train_mode)
                param.stop_gradient = not should_train
                if should_train:
                    train_params.append(param)
                    train_names.append(f"L{layer_i}.{name}")
        print(f"trainable params: {len(train_params)} ({sum(int(p.numel().item()) for p in train_params)} elems)", flush=True)
        print(f"first trainable: {train_names[:4]}...{train_names[-4:]}", flush=True)
        optimizer = paddle.optimizer.AdamW(learning_rate=args.lr, beta1=0.9, beta2=0.95, weight_decay=args.weight_decay, parameters=train_params, multi_precision=True)
        best_loss = None
        best_state = None
        initial_loss = None
        warmup_steps = max(int(args.refine_steps * args.warmup_ratio), 1)
        min_lr = args.lr * args.min_lr_ratio
        for step in range(args.refine_steps):
            if step < warmup_steps:
                current_lr = min_lr + (args.lr - min_lr) * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(args.refine_steps - warmup_steps, 1)
                current_lr = min_lr + 0.5 * (args.lr - min_lr) * (1.0 + np.cos(np.pi * progress))
            optimizer.set_lr(float(current_lr))
            indices = np.random.choice(gqa_logits_np.shape[0], min(args.batch_size, gqa_logits_np.shape[0]), replace=False)
            input_ids = paddle.to_tensor(token_ids_all[indices].astype(np.int64))
            target_logits = paddle.to_tensor(gqa_logits_np[indices])
            vha_x = vha_model.run_function[0]({"input_ids": input_ids})
            vha_x = run_layers(vha_model, vha_layers, vha_x, 0, num_layers - 1)
            for fn_idx in range(vha_layers[-1] + 1, len(vha_model.run_function)):
                vha_x = vha_model.run_function[fn_idx](vha_x)
            student_logits = (vha_x if isinstance(vha_x, paddle.Tensor) else vha_x["hidden_states"]).cast("float32")
            teacher_lp = F.log_softmax(target_logits, axis=-1)
            student_lp = F.log_softmax(student_logits, axis=-1)
            kl = paddle.sum(paddle.exp(teacher_lp) * (teacher_lp - student_lp), axis=-1)
            loss = paddle.mean(kl)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm_sq = None
                for p in train_params:
                    if p.grad is not None:
                        grad_norm_sq = paddle.sum(paddle.square(p.grad.cast("float32"))) if grad_norm_sq is None else grad_norm_sq + paddle.sum(paddle.square(p.grad.cast("float32")))
                if grad_norm_sq is not None:
                    clip_coef = args.grad_clip / (paddle.sqrt(grad_norm_sq) + 1e-6)
                    if float(clip_coef.numpy()) < 1.0:
                        for p in train_params:
                            if p.grad is not None:
                                p.grad = p.grad * clip_coef.cast(p.grad.dtype)
            optimizer.step()
            optimizer.clear_grad()
            loss_val = float(loss.numpy())
            if initial_loss is None:
                initial_loss = loss_val
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_state = [p.numpy().copy() for p in train_params]
            if step % 5 == 0 or step == args.refine_steps - 1:
                reduction = 0.0 if initial_loss is None else (1.0 - loss_val / initial_loss) * 100.0
                print(f"step={step} kl={loss_val:.6f} best={best_loss:.6f} red={reduction:.2f}% lr={current_lr:.8g}", flush=True)
        if best_state is not None:
            for p, v in zip(train_params, best_state):
                p.set_value(paddle.to_tensor(v).cast(p.dtype))
        records = [{"mode": "logits_kl", "layers": f"{first_endpoint}-{endpoint_limit}", "initial_loss": initial_loss, "best_loss": best_loss}]
        print("\nSaving refined checkpoint", flush=True)
        save_pipeline_model(vha_model, args.output_path, os.path.join(args.vha_checkpoint, "config.json"))
        (Path(args.output_path) / "refine_alignment_metrics.json").write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
        print(f"Done: {args.output_path}", flush=True)
        return

    records = []
    for endpoint in endpoints:
        start_layer = max(0, endpoint - args.train_window + 1)
        print(f"\n=== refine endpoint={endpoint} window={start_layer}..{endpoint} mode={args.train_mode} ===", flush=True)

        inputs = []
        targets = []
        with paddle.no_grad():
            for batch_start in range(0, token_ids.shape[0], args.batch_size):
                batch = token_ids[batch_start:batch_start + args.batch_size]
                input_ids = paddle.to_tensor(batch.astype(np.int64))
                gqa_x = gqa_model.run_function[0]({"input_ids": input_ids})
                gqa_x = run_layers(gqa_model, gqa_layers, gqa_x, 0, endpoint)
                targets.append(gqa_x["hidden_states"].detach().cast("float32").numpy())

                vha_x = vha_model.run_function[0]({"input_ids": input_ids})
                if start_layer > 0:
                    vha_x = run_layers(vha_model, vha_layers, vha_x, 0, start_layer - 1)
                inputs.append(vha_x["hidden_states"].detach().cast("float32").numpy())
        input_np = np.concatenate(inputs, axis=0)
        target_np = np.concatenate(targets, axis=0)
        del inputs, targets
        paddle.device.cuda.empty_cache()

        train_params = []
        train_names = []
        for layer_i, tl_idx in enumerate(vha_layers):
            layer = vha_model.run_function[tl_idx]
            layer_in_window = start_layer <= layer_i <= endpoint
            for name, param in layer.named_parameters():
                should_train = layer_in_window and is_trainable_param(name, args.train_mode)
                param.stop_gradient = not should_train
                if should_train:
                    train_params.append(param)
                    train_names.append(f"L{layer_i}.{name}")
        print(f"trainable params: {len(train_params)} ({sum(int(p.numel().item()) for p in train_params)} elems)", flush=True)
        print(f"first trainable: {train_names[:8]}", flush=True)
        if args.eval_only or not train_params:
            continue

        optimizer = paddle.optimizer.AdamW(
            learning_rate=args.lr,
            beta1=0.9,
            beta2=0.95,
            weight_decay=args.weight_decay,
            parameters=train_params,
            multi_precision=True,
        )
        best_loss = None
        best_state = None
        initial_loss = None
        warmup_steps = max(int(args.refine_steps * args.warmup_ratio), 1)
        min_lr = args.lr * args.min_lr_ratio
        for step in range(args.refine_steps):
            if step < warmup_steps:
                current_lr = min_lr + (args.lr - min_lr) * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(args.refine_steps - warmup_steps, 1)
                current_lr = min_lr + 0.5 * (args.lr - min_lr) * (1.0 + np.cos(np.pi * progress))
            optimizer.set_lr(float(current_lr))
            indices = np.random.choice(input_np.shape[0], min(args.batch_size, input_np.shape[0]), replace=False)
            x_batch = paddle.to_tensor(input_np[indices]).cast("bfloat16")
            y_batch = paddle.to_tensor(target_np[indices]).cast("float32")
            rpe = paddle.to_tensor(rotary_pos_emb).cast("bfloat16")
            x_dict = {"hidden_states": x_batch, "rotary_pos_emb": rpe}
            x_dict = run_layers(vha_model, vha_layers, x_dict, start_layer, endpoint)
            pred = x_dict["hidden_states"].cast("float32")
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm_sq = None
                for param in train_params:
                    if param.grad is not None:
                        grad_sq = paddle.sum(paddle.square(param.grad.cast("float32")))
                        grad_norm_sq = grad_sq if grad_norm_sq is None else grad_norm_sq + grad_sq
                if grad_norm_sq is not None:
                    grad_norm = paddle.sqrt(grad_norm_sq)
                    clip_coef = args.grad_clip / (grad_norm + 1e-6)
                    if float(clip_coef.numpy()) < 1.0:
                        for param in train_params:
                            if param.grad is not None:
                                param.grad = param.grad * clip_coef.cast(param.grad.dtype)
            optimizer.step()
            optimizer.clear_grad()
            loss_val = float(loss.numpy())
            if initial_loss is None:
                initial_loss = loss_val
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_state = [param.numpy().copy() for param in train_params]
            if step % 10 == 0 or step == args.refine_steps - 1:
                reduction = 0.0 if initial_loss is None else (1.0 - loss_val / initial_loss) * 100.0
                print(f"step={step} loss={loss_val:.8f} best={best_loss:.8f} reduction={reduction:.2f}% lr={current_lr:.8g}", flush=True)
        if best_state is not None:
            for param, value in zip(train_params, best_state):
                param.set_value(paddle.to_tensor(value).cast(param.dtype))
        for param in train_params:
            param.stop_gradient = True
        records.append({"endpoint": endpoint, "start_layer": start_layer, "initial_loss": initial_loss, "best_loss": best_loss})
        del optimizer, input_np, target_np, best_state
        gc.collect()
        paddle.device.cuda.empty_cache()

    print("\nSaving refined checkpoint", flush=True)
    save_pipeline_model(vha_model, args.output_path, os.path.join(args.vha_checkpoint, "config.json"))
    (Path(args.output_path) / "refine_alignment_metrics.json").write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
    print(f"Done: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
