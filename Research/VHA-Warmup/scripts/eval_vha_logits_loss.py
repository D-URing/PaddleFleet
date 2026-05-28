#!/usr/bin/env python3
"""Evaluate GQA teacher vs VHA student with CE and logits KL on real tokens."""

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
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")

    loaded = 0
    skipped = []
    for sf in sf_files:
        with safe_open(os.path.join(checkpoint_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                value = f.get_tensor(key).float().numpy()
                candidates = []
                pipeline_name = ckpt_key_to_pipeline_name(key, num_layers)
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
                    skipped.append(key if target_name is None else f"{key}->{target_name}: shape {value.shape} vs {list(target.shape)}")
    return loaded, len(param_dict), skipped


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
    return np.random.randint(0, 151936, (num_samples, seq_length), dtype=np.int64)


def as_logits(output):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        for key in ("logits", "lm_logits", "output"):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    return output


def compute_ce(logits, labels):
    import paddle.nn.functional as F

    vocab_size = logits.shape[-1]
    return float(F.cross_entropy(logits.reshape([-1, vocab_size]).cast("float32"), labels.reshape([-1]), reduction="mean").numpy())


def compute_forward_kl(teacher_logits, student_logits, temperature=1.0, chunk_size=128):
    import paddle
    import paddle.nn.functional as F

    seq_len = min(teacher_logits.shape[1], student_logits.shape[1])
    total = paddle.zeros([1], dtype="float32")
    count = 0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        teacher_chunk = teacher_logits[:, start:end, :].cast("float32") / temperature
        student_chunk = student_logits[:, start:end, :].cast("float32") / temperature
        teacher_probs = F.softmax(teacher_chunk, axis=-1)
        student_log_probs = F.log_softmax(student_chunk, axis=-1)
        teacher_log_probs = F.log_softmax(teacher_chunk, axis=-1)
        kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1)
        total += kl.sum() * (temperature * temperature)
        count += int(np.prod(kl.shape))
    return float((total / max(count, 1)).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_checkpoint", required=True)
    parser.add_argument("--vha_checkpoint", required=True)
    parser.add_argument("--gqa_model_config", required=True)
    parser.add_argument("--vha_model_config", required=True)
    parser.add_argument("--data_path", default=None, help="mmap basename without .bin/.idx")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    import paddle

    use_cuda = paddle.device.is_compiled_with_cuda()
    if use_cuda:
        paddle.set_device("gpu:0")
    setup_paddlefleet_single_gpu()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "VHA"))
    from models.qwen_provider import create_provider

    def build_model(config_path, checkpoint_path, label):
        provider = create_provider(config_path)
        provider.seq_length = args.seq_length
        provider.max_sequence_length = args.seq_length
        model = provider.provide()
        if hasattr(model, "to") and use_cuda:
            model = model.to(device="gpu")
        loaded, total, skipped = load_safetensors_into_pipeline_model(model, checkpoint_path)
        print(f"Loaded {loaded}/{total} {label} params; skipped={len(skipped)}", flush=True)
        if skipped:
            print(f"  first skipped {label}: {skipped[:5]}", flush=True)
        model.eval()
        for param in model.parameters():
            param.stop_gradient = True
        return model

    gqa_model = build_model(args.gqa_model_config, args.gqa_checkpoint, "GQA")
    vha_model = build_model(args.vha_model_config, args.vha_checkpoint, "VHA")
    token_ids = load_tokens(args.data_path, args.num_samples, args.seq_length)

    rows = []
    teacher_ce_values = []
    student_ce_values = []
    kl_values = []
    place = paddle.CUDAPlace(0) if use_cuda else None
    with paddle.no_grad():
        for batch_start in range(0, token_ids.shape[0], args.batch_size):
            batch = token_ids[batch_start:batch_start + args.batch_size]
            input_ids = paddle.to_tensor(batch[:, :-1].astype(np.int64), place=place)
            labels = paddle.to_tensor(batch[:, 1:].astype(np.int64), place=place)

            teacher_logits = as_logits(gqa_model({"input_ids": input_ids}))
            student_logits = as_logits(vha_model({"input_ids": input_ids}))
            min_len = min(teacher_logits.shape[1], student_logits.shape[1], labels.shape[1])
            teacher_logits = teacher_logits[:, :min_len, :]
            student_logits = student_logits[:, :min_len, :]
            labels = labels[:, :min_len]

            teacher_ce = compute_ce(teacher_logits, labels)
            student_ce = compute_ce(student_logits, labels)
            kl = compute_forward_kl(teacher_logits, student_logits, temperature=args.kl_temperature)
            row = {"batch_start": batch_start, "teacher_ce": teacher_ce, "student_ce": student_ce, "ce_gap": student_ce - teacher_ce, "teacher_to_student_kl": kl}
            rows.append(row)
            teacher_ce_values.append(teacher_ce)
            student_ce_values.append(student_ce)
            kl_values.append(kl)
            print(f"batch {batch_start}: teacher_ce={teacher_ce:.6f}, student_ce={student_ce:.6f}, gap={student_ce - teacher_ce:.6f}, KL={kl:.6f}", flush=True)

    summary = {
        "num_samples": int(token_ids.shape[0]),
        "seq_length": args.seq_length,
        "teacher_ce": float(np.mean(teacher_ce_values)),
        "student_ce": float(np.mean(student_ce_values)),
        "ce_gap": float(np.mean(student_ce_values) - np.mean(teacher_ce_values)),
        "teacher_to_student_kl": float(np.mean(kl_values)),
        "batches": rows,
    }
    print("mean: teacher_ce={teacher_ce:.6f}, student_ce={student_ce:.6f}, gap={ce_gap:.6f}, KL={teacher_to_student_kl:.6f}".format(**summary), flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
