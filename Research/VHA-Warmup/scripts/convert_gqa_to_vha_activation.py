#!/usr/bin/env python3
"""
Activation-based GQA -> VHA conversion.

Uses calibration data to find optimal KV compression and Q+premix initialization
via PCA on activations (TransMLA-inspired) + alternating optimization.

Steps:
  1. Collect Q/K/V activations from GQA model on calibration data
  2. KV compression via joint activation PCA (8 heads -> 2 heads)
  3. Q + premix joint optimization (alternating least-squares)
  4. Postmix initialization (low-rank V-residual compensation)
  5. Assemble and save VHA checkpoint

Usage:
    python convert_gqa_to_vha_activation.py \
        --gqa_checkpoint ./output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged \
        --output_path ./output/qwen3_vha_1p7B_init_activation \
        --calib_data ../../../datasets/fineweb-edu \
        --num_calib_samples 512 \
        --seq_length 4096
"""

import argparse
import gc
import json
import os
import pickle
import sys

import numpy as np
import paddle
from safetensors import safe_open
from safetensors.numpy import save_file


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


# =============================================================================
# Config
# =============================================================================

D = 2048          # hidden_size
d = 128           # head_dim
SRC_H_Q = 16     # source Q heads
SRC_H_K = 8      # source KV heads
TGT_H_Q = 8      # target Q heads for full VHA premix mode
KV_ONLY_H_Q = SRC_H_Q  # target Q heads for KV-only mode
TGT_H_K = 2      # target KV heads
POSTMIX_RANK = 4
NUM_LAYERS = 28
JOINT_OPT_ITERS = 5
USE_GPU_LINALG = True
_GPU_DEVICE_INITIALIZED = False


def _gpu_available():
    return USE_GPU_LINALG and paddle.device.is_compiled_with_cuda()


def _ensure_gpu_device():
    global _GPU_DEVICE_INITIALIZED
    if _gpu_available() and not _GPU_DEVICE_INITIALIZED:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        paddle.set_device("gpu")  # Auto-select from CUDA_VISIBLE_DEVICES
        print(f"  Using Paddle GPU linalg on {paddle.device.get_device()} (CUDA_VISIBLE_DEVICES={visible_devices})", flush=True)
        _GPU_DEVICE_INITIALIZED = True


def _to_gpu(array):
    _ensure_gpu_device()
    return paddle.to_tensor(array, dtype="float32", place=paddle.CUDAPlace(0))


def _to_numpy(tensor):
    return tensor.numpy().astype(np.float32)


def save_activation_cache(activations, cache_path, meta):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    arrays = {}
    for layer_idx, layer_acts in activations.items():
        for name, value in layer_acts.items():
            if value is not None:
                arrays[f"layer{layer_idx}.{name}"] = value.astype(np.float16)
    arrays["__meta__"] = np.array(json.dumps(meta), dtype=np.unicode_)
    np.savez(cache_path, **arrays)  # No compression for speed
    print(f"  Saved activation cache: {cache_path} ({len(arrays) - 1} tensors)", flush=True)


def load_activation_layer(cache_path, layer_idx, max_tokens=None):
    """Load one layer of Q/K/V/pre_o activations from a merged cache."""
    layer_acts = {}
    with np.load(cache_path, allow_pickle=False) as data:
        for name in ("Q", "K", "V", "pre_o"):
            key = f"layer{layer_idx}.{name}"
            if key in data.files:
                value = data[key]
                if max_tokens is not None:
                    value = value[:max_tokens]
                layer_acts[name] = value.astype(np.float32)
            elif name == "pre_o":
                layer_acts[name] = None
            else:
                raise KeyError(f"Missing required activation tensor: {key}")
    return layer_acts


# =============================================================================
# QKV grouped layout helpers
# =============================================================================

def split_grouped_qkv_lastdim(qkv, num_q_heads, num_kv_heads, head_dim):
    """Split PaddleFleet grouped QKV layout along the last dimension."""
    heads_per_group = num_q_heads // num_kv_heads
    group_dim = (heads_per_group + 2) * head_dim
    leading_shape = qkv.shape[:-1]
    grouped = qkv.reshape(*leading_shape, num_kv_heads, group_dim)
    q_dim = heads_per_group * head_dim
    q = grouped[..., :q_dim].reshape(*leading_shape, num_q_heads * head_dim)
    k = grouped[..., q_dim:q_dim + head_dim].reshape(*leading_shape, num_kv_heads * head_dim)
    v = grouped[..., q_dim + head_dim:].reshape(*leading_shape, num_kv_heads * head_dim)
    return q, k, v


def pack_grouped_qkv_lastdim(q, k, v, num_q_heads, num_kv_heads, head_dim):
    """Pack Q/K/V tensors into PaddleFleet grouped QKV layout along last dim."""
    heads_per_group = num_q_heads // num_kv_heads
    leading_shape = q.shape[:-1]
    q_grouped = q.reshape(*leading_shape, num_kv_heads, heads_per_group * head_dim)
    k_grouped = k.reshape(*leading_shape, num_kv_heads, head_dim)
    v_grouped = v.reshape(*leading_shape, num_kv_heads, head_dim)
    grouped = np.concatenate([q_grouped, k_grouped, v_grouped], axis=-1)
    return grouped.reshape(*leading_shape, num_kv_heads * (heads_per_group + 2) * head_dim)


def setup_paddlefleet_single_gpu():
    """Patch PaddleFleet helpers for standalone TP=PP=1 conversion."""
    import contextlib
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

    from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    def _simple_linear(input_, weight, bias=None):
        out_shape = list(input_.shape[:-1]) + [weight.shape[-1]]
        flat_input = input_.reshape([-1, input_.shape[-1]]).cast("float32")
        out = flat_input.matmul(weight.cast("float32")).cast(input_.dtype).reshape(out_shape)
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


# =============================================================================
# Step 0: Load GQA weights
# =============================================================================

def load_gqa_weights(checkpoint_dir):
    """Load GQA checkpoint, return per-layer Q/K/V/O weights as float32."""
    sf_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')])
    assert sf_files, f"No safetensors in {checkpoint_dir}"

    all_tensors = {}
    for sf in sf_files:
        with safe_open(os.path.join(checkpoint_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key).float().numpy()

    # Parse per-layer attention weights from fused qkv_proj
    layers = {}
    for layer_idx in range(NUM_LAYERS):
        prefix = f"layers.{layer_idx}.self_attn"
        qkv = all_tensors[f"{prefix}.qkv_proj.weight"]  # PaddleFleet grouped layout
        W_q, W_k, W_v = split_grouped_qkv_lastdim(qkv, SRC_H_Q, SRC_H_K, d)

        layers[layer_idx] = {
            "W_q": W_q,                                # [D, 2048]
            "W_k": W_k,                                # [D, 1024]
            "W_v": W_v,                                # [D, 1024]
            "W_o": all_tensors[f"{prefix}.o_proj.weight"],  # [2048, D]
            "q_norm": all_tensors.get(f"{prefix}.q_norm.weight"),
            "k_norm": all_tensors.get(f"{prefix}.k_norm.weight"),
        }

    # Non-attention weights
    other_tensors = {k: v for k, v in all_tensors.items()
                     if "self_attn.qkv_proj" not in k and "self_attn.o_proj" not in k}

    return layers, other_tensors


# =============================================================================
# Step 0b: Collect activations (simplified - weight-space proxy)
# =============================================================================

def collect_activations_real(layers, other_tensors, calib_data_path, num_samples, seq_length, args=None):
    """
    Collect Q/K/V activations using paddlefleet GQA model on GPU.
    
    Loads the GQA model via the same provider used in training,
    registers hooks on each TransformerLayer's self_attn to capture
    Q/K/V activations (pre-RoPE), then runs forward on calibration data.
    """
    import paddle
    setup_paddlefleet_single_gpu()
    paddle.set_device("gpu")  # Auto-select from CUDA_VISIBLE_DEVICES
    
    # Add parent path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'VHA'))
    from models.qwen_provider import create_provider
    from paddleformers.data.causal_dataset import build_train_valid_test_datasets
    
    print(f"Collecting activations via paddlefleet GPU forward (N={num_samples}, S={seq_length})...")
    
    # Build GQA model
    print(f"  Loading GQA model from config: {args.gqa_model_config}")
    provider = create_provider(args.gqa_model_config)
    provider.seq_length = seq_length
    provider.max_sequence_length = seq_length
    model = provider.provide()
    model = model.to(device="gpu")
    
    # Load checkpoint weights
    print(f"  Loading GQA checkpoint weights...")
    from safetensors import safe_open
    gqa_ckpt_dir = args.gqa_checkpoint
    print(f"  Loading weights from: {gqa_ckpt_dir}")
    sf_files = sorted([f for f in os.listdir(gqa_ckpt_dir) if f.endswith('.safetensors')])
    
    state_dict = {}
    for sf in sf_files:
        with safe_open(os.path.join(gqa_ckpt_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).float().numpy()
    
    # Load into model (handle key prefix)
    model_state = model.state_dict()
    for ckpt_key, ckpt_val in state_dict.items():
        candidates = [ckpt_key, "model." + ckpt_key]
        for c in candidates:
            if c in model_state:
                param = dict(model.named_parameters()).get(c) or dict(model.named_buffers()).get(c)
                if param is not None and list(param.shape) == list(ckpt_val.shape):
                    param.set_value(paddle.to_tensor(ckpt_val, place=paddle.CUDAPlace(0)).cast(param.dtype))
                break
    
    model.eval()
    for p in model.parameters():
        p.stop_gradient = True
    print(f"  GQA model loaded on GPU")
    
    # Register hooks to capture Q/K/V activations (pre-RoPE)
    # Find qkv_proj modules via named_sublayers (works with PipelineLayer)
    qkv_modules = []
    o_proj_modules = []
    for name, module in model.named_sublayers():
        if 'qkv_proj' in name:
            qkv_modules.append((name, module))
        elif 'o_proj' in name:
            o_proj_modules.append((name, module))
    print(f"  Found {len(qkv_modules)} qkv_proj modules for hooking")
    print(f"  Found {len(o_proj_modules)} o_proj modules for pre-O hooks")
    
    captured_qkv = {}  # layer_idx -> Paddle tensor on GPU
    captured_pre_o = {}  # layer_idx -> Paddle tensor on GPU before o_proj
    hooks = []
    
    for idx, (name, qkv_module) in enumerate(qkv_modules):
        def make_qkv_hook(layer_idx):
            def hook_fn(module, input, output):
                captured_qkv[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach()
            return hook_fn
        
        h = qkv_module.register_forward_post_hook(make_qkv_hook(idx))
        hooks.append(h)

    for idx, (name, o_proj_module) in enumerate(o_proj_modules):
        def make_pre_o_hook(layer_idx):
            def hook_fn(module, input):
                captured_pre_o[layer_idx] = input[0].detach()
            return hook_fn

        h = o_proj_module.register_forward_pre_hook(make_pre_o_hook(idx))
        hooks.append(h)
    
    # Load calibration data
    print(f"  Loading calibration data...")
    if calib_data_path is not None:
        import glob as glob_mod
        idx_files = glob_mod.glob(os.path.join(calib_data_path, "*_idx.npz")) + \
                    glob_mod.glob(os.path.join(calib_data_path, "*.idx"))
        data_files = [f.replace("_idx.npz", "").replace(".idx", "") for f in idx_files]
        
        if data_files:
            if len(data_files) == 1:
                data_prefix = [data_files[0]]
            else:
                data_prefix = []
                for data_file in data_files:
                    data_prefix.extend([1.0, data_file])
            print(f"  Data prefix: {data_prefix}", flush=True)
            try:
                train_ds, _, _ = build_train_valid_test_datasets(
                    data_prefix=data_prefix,
                    data_impl="mmap",
                    splits_string="1,0,0",
                    train_val_test_num_samples=[num_samples, 0, 0],
                    seq_length=seq_length,
                    seed=42,
                    skip_warmup=True,
                    share_folder=False,
                    data_cache_path=None,
                    need_data=True,
                )
                token_ids = []
                for i in range(min(num_samples, len(train_ds))):
                    token_ids.append(train_ds[i]["text"][:seq_length])
                token_ids = np.stack(token_ids)
                print(f"  Loaded {token_ids.shape[0]} samples, seq_len={token_ids.shape[1]}")
            except BaseException as e:
                print(f"  WARNING: mmap load failed: {type(e).__name__}: {e!r}, using random tokens")
                token_ids = np.random.randint(0, 151936, (num_samples, seq_length))
        else:
            token_ids = np.random.randint(0, 151936, (num_samples, seq_length))
    else:
        token_ids = np.random.randint(0, 151936, (num_samples, seq_length))

    # Sharding handled at launch level: each rank loads its own per-rank num_samples
    # No need to slice the already-loaded data
    if args is not None and args.activation_shard_count > 1:
        shard_rank = args.activation_shard_rank
        shard_count = args.activation_shard_count
        # Each rank loads its own num_samples=per_rank samples directly
        # No slicing needed - data loading already distributes work
        print(f"  Activation shard {shard_rank}/{shard_count}: {num_samples} samples (loaded directly)", flush=True)
    
    # Run forward pass in batches, collect activations per layer
    batch_size = args.activation_batch_size if args is not None else 1
    max_tokens_per_layer = args.max_calib_tokens if args is not None else 20000
    
    # We need to collect activations layer by layer
    # But hooks fire for ALL layers in one forward pass
    # So we run forward once per batch and collect all layers at once
    
    layer_Q_accum = {i: [] for i in range(NUM_LAYERS)}
    layer_K_accum = {i: [] for i in range(NUM_LAYERS)}
    layer_V_accum = {i: [] for i in range(NUM_LAYERS)}
    layer_pre_o_accum = {i: [] for i in range(NUM_LAYERS)}
    tokens_collected = 0
    
    print(f"  Running forward pass (batch_size={batch_size}, device={paddle.device.get_device()})...", flush=True)
    
    with paddle.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            if tokens_collected >= max_tokens_per_layer:
                break
            
            batch_end = min(batch_start + batch_size, num_samples)
            batch_tokens = token_ids[batch_start:batch_end]
            
            input_ids = paddle.to_tensor(batch_tokens.astype(np.int64), place=paddle.CUDAPlace(0))
            
            # Clear captured
            captured_qkv.clear()
            captured_pre_o.clear()
            
            batch_idx = batch_start // batch_size
            print(f"    Batch {batch_idx}: forward start ({batch_end - batch_start} samples)", flush=True)

            # Forward pass (PipelineLayer for pp=1 just runs all layers)
            try:
                _ = model({"input_ids": input_ids})
            except Exception as e:
                print(f"  Forward pass error: {e}")
                print(f"  Trying alternative input format...")
                try:
                    _ = model(input_ids)
                except Exception as e2:
                    print(f"  Also failed: {e2}")
                    break
            print(f"    Batch {batch_idx}: forward done, processing activations", flush=True)
            
            # Process captured QKV for each layer
            for layer_idx in range(NUM_LAYERS):
                if layer_idx not in captured_qkv:
                    continue
                
                qkv_raw = captured_qkv[layer_idx]  # [B, S, qkv_dim] or [B, S, ng, hpg*d+2*d]
                
                # Flatten to [B*S, qkv_dim] on GPU
                if qkv_raw.ndim == 4:
                    B, S_len, ng, feat = qkv_raw.shape
                    qkv_flat = qkv_raw.reshape([B * S_len, ng * feat])
                else:
                    B, S_len, feat = qkv_raw.shape
                    qkv_flat = qkv_raw.reshape([B * S_len, feat])
                
                # Split grouped QKV into Q, K, V on GPU before copying to CPU
                heads_per_group = SRC_H_Q // SRC_H_K
                group_dim = (heads_per_group + 2) * d
                qkv_grouped = qkv_flat.reshape([qkv_flat.shape[0], SRC_H_K, group_dim])
                q_dim = heads_per_group * d
                Q_batch = qkv_grouped[:, :, :q_dim].reshape([qkv_flat.shape[0], SRC_H_Q * d])
                K_batch = qkv_grouped[:, :, q_dim:q_dim + d].reshape([qkv_flat.shape[0], SRC_H_K * d])
                V_batch = qkv_grouped[:, :, q_dim + d:].reshape([qkv_flat.shape[0], SRC_H_K * d])
                
                # Subsample then copy only retained tensors to CPU
                keep = min(Q_batch.shape[0], max_tokens_per_layer - tokens_collected)
                layer_Q_accum[layer_idx].append(Q_batch[:keep].cast("float32").numpy())
                layer_K_accum[layer_idx].append(K_batch[:keep].cast("float32").numpy())
                layer_V_accum[layer_idx].append(V_batch[:keep].cast("float32").numpy())
                if layer_idx in captured_pre_o:
                    pre_o_raw = captured_pre_o[layer_idx]
                    if pre_o_raw.ndim == 4:
                        pre_o_flat = pre_o_raw.reshape([pre_o_raw.shape[0] * pre_o_raw.shape[1], -1])
                    else:
                        pre_o_flat = pre_o_raw.reshape([pre_o_raw.shape[0] * pre_o_raw.shape[1], pre_o_raw.shape[-1]])
                    layer_pre_o_accum[layer_idx].append(pre_o_flat[:keep].cast("float32").numpy())
            
            tokens_collected += min(
                batch_tokens.shape[0] * batch_tokens.shape[1],
                max_tokens_per_layer - tokens_collected,
            )
            
            if (batch_start // batch_size) % 8 == 0:
                print(f"    Batch {batch_start//batch_size}: {tokens_collected} tokens collected")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Assemble activations
    activations = {}
    for layer_idx in range(NUM_LAYERS):
        if layer_Q_accum[layer_idx]:
            Q_act = np.concatenate(layer_Q_accum[layer_idx], axis=0)
            K_act = np.concatenate(layer_K_accum[layer_idx], axis=0)
            V_act = np.concatenate(layer_V_accum[layer_idx], axis=0)
            pre_o_act = np.concatenate(layer_pre_o_accum[layer_idx], axis=0) if layer_pre_o_accum[layer_idx] else None
        else:
            # Fallback: use weight-based proxy
            print(f"  WARNING: No activations for layer {layer_idx}, using proxy")
            np.random.seed(42 + layer_idx)
            x = np.random.randn(10000, D).astype(np.float32) * 0.02
            Q_act = x @ layers[layer_idx]["W_q"]
            K_act = x @ layers[layer_idx]["W_k"]
            V_act = x @ layers[layer_idx]["W_v"]
            pre_o_act = None
        
        activations[layer_idx] = {"Q": Q_act, "K": K_act, "V": V_act, "pre_o": pre_o_act}
        
        if layer_idx % 7 == 0:
            print(f"  Layer {layer_idx}: Q={Q_act.shape}, K_norm={np.linalg.norm(K_act)/np.sqrt(K_act.size):.4f}")
    
    # Cleanup GPU
    del model
    paddle.device.cuda.empty_cache()
    gc.collect()
    
    return activations


def compute_kv_reconstruction_error(K_act, V_act, P_k, P_v):
    """Compute relative reconstruction errors for K/V projection matrices."""
    if _gpu_available():
        K_t = _to_gpu(K_act)
        V_t = _to_gpu(V_act)
        Pk_t = _to_gpu(P_k)
        Pv_t = _to_gpu(P_v)
        K_recon = paddle.matmul(paddle.matmul(K_t, Pk_t), Pk_t, transpose_y=True)
        V_recon = paddle.matmul(paddle.matmul(V_t, Pv_t), Pv_t, transpose_y=True)
        k_error = paddle.linalg.norm(K_t - K_recon) / (paddle.linalg.norm(K_t) + 1e-8)
        v_error = paddle.linalg.norm(V_t - V_recon) / (paddle.linalg.norm(V_t) + 1e-8)
        return float(k_error.numpy()), float(v_error.numpy())

    K_compressed = K_act @ P_k
    K_recon = K_compressed @ P_k.T
    k_error = np.linalg.norm(K_act - K_recon) / (np.linalg.norm(K_act) + 1e-8)

    V_compressed = V_act @ P_v
    V_recon = V_compressed @ P_v.T
    v_error = np.linalg.norm(V_act - V_recon) / (np.linalg.norm(V_act) + 1e-8)
    return float(k_error), float(v_error)


def balanced_pca_basis(K_group, V_group, target_dim):
    """Joint balanced PCA basis for a K/V group."""
    N = K_group.shape[0]
    source_dim = K_group.shape[1]
    if _gpu_available():
        K_t = _to_gpu(K_group)
        V_t = _to_gpu(V_group)
        scale_k_t = paddle.linalg.norm(V_t) / (paddle.linalg.norm(K_t) + 1e-8)
        KV_joint = paddle.concat([K_t * scale_k_t, V_t], axis=1)
        C = paddle.matmul(KV_joint, KV_joint, transpose_x=True) / N
        _, eigenvectors = paddle.linalg.eigh(C)
        P_joint = eigenvectors[:, -target_dim:]
        Pk_group = P_joint[:source_dim, :] / scale_k_t
        Pv_group = P_joint[source_dim:, :]
        return _to_numpy(Pk_group), _to_numpy(Pv_group), float(scale_k_t.numpy())

    k_norm = np.linalg.norm(K_group)
    v_norm = np.linalg.norm(V_group)
    scale_k = v_norm / (k_norm + 1e-8)
    KV_joint = np.concatenate([K_group * scale_k, V_group], axis=1)
    C = KV_joint.T @ KV_joint / N
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    P_joint = eigenvectors[:, -target_dim:]
    Pk_group = P_joint[:source_dim, :] / scale_k
    Pv_group = P_joint[source_dim:, :]
    return Pk_group.astype(np.float32), Pv_group.astype(np.float32), float(scale_k)


def find_best_kv_partition(K_act, V_act, Q_act):
    """Search all balanced 8->2 partitions and choose the lowest fusion loss."""
    import itertools

    N = K_act.shape[0]
    heads = list(range(SRC_H_K))
    q_per_kv = SRC_H_Q // SRC_H_K
    best = None

    if _gpu_available():
        K_heads_t = _to_gpu(K_act).reshape([N, SRC_H_K, d])
        V_heads_t = _to_gpu(V_act).reshape([N, SRC_H_K, d])
        Q_heads_t = _to_gpu(Q_act).reshape([N, SRC_H_Q, d])
        target_t = _to_gpu(build_virtual_v_targets(V_act)).reshape([N, TGT_H_K, TGT_H_Q, d])

        for first_group in itertools.combinations(heads[1:], SRC_H_K // TGT_H_K - 1):
            group0 = tuple(sorted((0,) + first_group))
            group1 = tuple(h for h in heads if h not in group0)
            partition = [group0, group1]
            loss = 0.0
            group_info = []
            for group_idx, group_heads in enumerate(partition):
                index_t = paddle.to_tensor(group_heads, dtype="int64", place=paddle.CUDAPlace(0))
                K_group_t = paddle.index_select(K_heads_t, index_t, axis=1).reshape([N, -1])
                V_group_t = paddle.index_select(V_heads_t, index_t, axis=1).reshape([N, -1])
                K_group = _to_numpy(K_group_t)
                V_group = _to_numpy(V_group_t)
                Pk_group, Pv_group, scale_k = balanced_pca_basis(K_group, V_group, d)
                Pk_t = _to_gpu(Pk_group)
                Pv_t = _to_gpu(Pv_group)
                K_new_t = paddle.matmul(K_group_t, Pk_t)
                V_new_t = paddle.matmul(V_group_t, Pv_t)
                K_recon_t = paddle.matmul(K_new_t, Pk_t, transpose_y=True)
                V_recon_t = paddle.matmul(V_new_t, Pv_t, transpose_y=True)
                recon_loss_t = (
                    paddle.mean((K_group_t - K_recon_t) ** 2) / (paddle.mean(K_group_t ** 2) + 1e-8)
                    + paddle.mean((V_group_t - V_recon_t) ** 2) / (paddle.mean(V_group_t ** 2) + 1e-8)
                )
                pred_v_t = paddle.tile(V_new_t.unsqueeze(1), [1, TGT_H_Q, 1])
                ref_v_t = target_t[:, group_idx, :, :]
                value_loss_t = paddle.mean((pred_v_t - ref_v_t) ** 2) / (paddle.mean(ref_v_t ** 2) + 1e-8)
                logit_sse_t = paddle.zeros([], dtype="float32")
                logit_ref_t = paddle.zeros([], dtype="float32")
                for local_q in range(TGT_H_Q):
                    virtual_head = group_idx * TGT_H_Q + local_q
                    source_kv_head = virtual_head // q_per_kv
                    ref_logit_t = paddle.sum(Q_heads_t[:, virtual_head, :] * K_heads_t[:, source_kv_head, :], axis=-1)
                    pred_logit_t = paddle.sum(Q_heads_t[:, virtual_head, :] * K_new_t, axis=-1)
                    logit_sse_t += paddle.sum((pred_logit_t - ref_logit_t) ** 2)
                    logit_ref_t += paddle.sum(ref_logit_t ** 2)
                logit_loss_t = logit_sse_t / (logit_ref_t + 1e-8)
                group_loss_t = logit_loss_t + 0.5 * value_loss_t + 0.1 * recon_loss_t
                group_loss = float(group_loss_t.numpy())
                loss += group_loss
                group_info.append({
                    "heads": [int(h) for h in group_heads],
                    "search_loss": group_loss,
                    "logit_loss": float(logit_loss_t.numpy()),
                    "value_loss": float(value_loss_t.numpy()),
                    "recon_loss": float(recon_loss_t.numpy()),
                    "scale_k": float(scale_k),
                })
            if best is None or loss < best[0]:
                best = (loss, partition, group_info)
        return [[int(h) for h in group] for group in best[1]], best[0], best[2]

    K_heads = K_act.reshape(N, SRC_H_K, d)
    V_heads = V_act.reshape(N, SRC_H_K, d)
    Q_heads = Q_act.reshape(N, SRC_H_Q, d)
    target = build_virtual_v_targets(V_act).reshape(N, TGT_H_K, TGT_H_Q, d)
    best = None

    for first_group in itertools.combinations(heads[1:], SRC_H_K // TGT_H_K - 1):
        group0 = tuple(sorted((0,) + first_group))
        group1 = tuple(h for h in heads if h not in group0)
        partition = [group0, group1]
        loss = 0.0
        group_info = []
        for group_idx, group_heads in enumerate(partition):
            K_group = K_heads[:, group_heads, :].reshape(N, -1)
            V_group = V_heads[:, group_heads, :].reshape(N, -1)
            Pk_group, Pv_group, scale_k = balanced_pca_basis(K_group, V_group, d)
            K_new = K_group @ Pk_group
            V_new = V_group @ Pv_group
            K_recon = K_new @ Pk_group.T
            V_recon = V_new @ Pv_group.T
            recon_loss = (
                np.mean((K_group - K_recon) ** 2) / (np.mean(K_group ** 2) + 1e-8)
                + np.mean((V_group - V_recon) ** 2) / (np.mean(V_group ** 2) + 1e-8)
            )
            pred_v = np.repeat(V_new[:, None, :], TGT_H_Q, axis=1)
            ref_v = target[:, group_idx, :, :]
            value_loss = np.mean((pred_v - ref_v) ** 2) / (np.mean(ref_v ** 2) + 1e-8)
            logit_sse = 0.0
            logit_ref = 0.0
            for local_q in range(TGT_H_Q):
                virtual_head = group_idx * TGT_H_Q + local_q
                source_kv_head = virtual_head // q_per_kv
                ref_logit = np.sum(Q_heads[:, virtual_head, :] * K_heads[:, source_kv_head, :], axis=-1)
                pred_logit = np.sum(Q_heads[:, virtual_head, :] * K_new, axis=-1)
                logit_sse += float(np.sum((pred_logit - ref_logit) ** 2))
                logit_ref += float(np.sum(ref_logit ** 2))
            logit_loss = logit_sse / (logit_ref + 1e-8)
            group_loss = logit_loss + 0.5 * value_loss + 0.1 * recon_loss
            loss += float(group_loss)
            group_info.append({
                "heads": [int(h) for h in group_heads],
                "search_loss": float(group_loss),
                "logit_loss": float(logit_loss),
                "value_loss": float(value_loss),
                "recon_loss": float(recon_loss),
                "scale_k": float(scale_k),
            })
        if best is None or loss < best[0]:
            best = (loss, partition, group_info)
    return [[int(h) for h in group] for group in best[1]], best[0], best[2]

def compress_kv_groupwise_pca(K_act, V_act, Q_act, blend_mean=0.1):
    """
    Compress KV by target groups: each target KV head fuses a local group of source KV heads.

    This is closer to DHA-style low-loss head fusion than global PCA: it preserves the
    coarse GQA grouping structure and learns one balanced joint PCA basis per group.
    """
    N = K_act.shape[0]
    merge_factor = SRC_H_K // TGT_H_K
    source_dim = merge_factor * d
    target_dim = d
    P_k = np.zeros((SRC_H_K * d, TGT_H_K * d), dtype=np.float32)
    P_v = np.zeros((SRC_H_K * d, TGT_H_K * d), dtype=np.float32)
    group_errors = []

    K_heads = K_act.reshape(N, SRC_H_K, d)
    V_heads = V_act.reshape(N, SRC_H_K, d)
    partition, search_loss_value, search_info = find_best_kv_partition(K_act, V_act, Q_act)
    k_groups = partition
    v_groups = partition
    search_diag = {"search_loss": float(search_loss_value), "groups": search_info}

    for group_idx in range(TGT_H_K):
        col_start = group_idx * d
        col_end = col_start + d
        k_src_heads = k_groups[group_idx]
        v_src_heads = v_groups[group_idx]

        K_group = K_heads[:, k_src_heads, :].reshape(N, source_dim)
        V_group = V_heads[:, v_src_heads, :].reshape(N, source_dim)

        Pk_group, Pv_group, scale_k = balanced_pca_basis(K_group, V_group, target_dim)

        if blend_mean > 0.0:
            mean_basis = np.zeros((source_dim, target_dim), dtype=np.float32)
            for head_idx in range(merge_factor):
                mean_basis[head_idx * d:(head_idx + 1) * d, :] = np.eye(d, dtype=np.float32) / merge_factor
            Pk_group = ((1.0 - blend_mean) * Pk_group + blend_mean * mean_basis).astype(np.float32)
            Pv_group = ((1.0 - blend_mean) * Pv_group + blend_mean * mean_basis).astype(np.float32)

        for local_idx, src_head in enumerate(k_src_heads):
            row_start = src_head * d
            row_end = row_start + d
            P_k[row_start:row_end, col_start:col_end] = Pk_group[local_idx * d:(local_idx + 1) * d, :]
        for local_idx, src_head in enumerate(v_src_heads):
            row_start = src_head * d
            row_end = row_start + d
            P_v[row_start:row_end, col_start:col_end] = Pv_group[local_idx * d:(local_idx + 1) * d, :]

        K_recon = (K_group @ Pk_group) @ Pk_group.T
        V_recon = (V_group @ Pv_group) @ Pv_group.T
        group_diag = {
            "group": group_idx,
            "k_src_heads": [int(x) for x in k_src_heads],
            "v_src_heads": [int(x) for x in v_src_heads],
            "k_error": float(np.linalg.norm(K_group - K_recon) / (np.linalg.norm(K_group) + 1e-8)),
            "v_error": float(np.linalg.norm(V_group - V_recon) / (np.linalg.norm(V_group) + 1e-8)),
            "scale_k": float(scale_k),
        }
        group_errors.append(group_diag)

    k_error, v_error = compute_kv_reconstruction_error(K_act, V_act, P_k, P_v)
    group_errors.append({"search_summary": search_diag})
    return P_k, P_v, k_error, v_error, group_errors


# =============================================================================
# Step 2: Q + premix joint optimization
# =============================================================================

def init_q_base_from_groups(Q_heads):
    """Initialize base Q heads as DHA-style averages over VHA target groups."""
    group0 = Q_heads[:, :TGT_H_Q, :]
    group1 = Q_heads[:, TGT_H_Q:, :]
    return ((group0 + group1) * 0.5).astype(np.float32)


def refine_kv_diagonal_attention(Q_act, K_act, V_act, P_q, premix, P_k, P_v):
    """Calibrate per-group K/V diagonal gains against the coupled QK/V proxy objective."""
    N = Q_act.shape[0]
    Q_heads = Q_act.reshape(N, SRC_H_Q, d)
    K_heads = K_act.reshape(N, SRC_H_K, d)
    V_target = build_virtual_v_targets(V_act)
    Q_base = (Q_act @ P_q).reshape(N, TGT_H_Q, d)
    K_new = (K_act @ P_k).reshape(N, TGT_H_K, d)
    V_new = (V_act @ P_v).reshape(N, TGT_H_K, d)
    refined_P_k = P_k.copy()
    refined_P_v = P_v.copy()
    q_per_kv = SRC_H_Q // SRC_H_K
    diag_info = []

    for group_idx in range(TGT_H_K):
        Q_vha = np.einsum("nqd,de->nqe", Q_base, premix[group_idx])
        K_group = K_new[:, group_idx, :]
        logit_features = []
        logit_targets = []
        for q_idx in range(TGT_H_Q):
            virtual_head = group_idx * TGT_H_Q + q_idx
            source_kv_head = virtual_head // q_per_kv
            logit_features.append(Q_vha[:, q_idx, :] * K_group)
            logit_targets.append(np.sum(Q_heads[:, virtual_head, :] * K_heads[:, source_kv_head, :], axis=-1))
        A_logit = np.concatenate(logit_features, axis=0)
        y_logit = np.concatenate(logit_targets, axis=0)
        k_scale = np.linalg.solve(
            A_logit.T @ A_logit + 1e-5 * np.eye(d, dtype=np.float32),
            A_logit.T @ y_logit,
        ).astype(np.float32)
        k_scale = np.clip(k_scale, 0.25, 4.0)

        V_group = V_new[:, group_idx, :]
        ref_v = V_target[:, group_idx * TGT_H_Q:(group_idx + 1) * TGT_H_Q, :]
        x_v = np.repeat(V_group[:, None, :], TGT_H_Q, axis=1)
        v_scale = (np.sum(x_v * ref_v, axis=(0, 1)) / (np.sum(x_v * x_v, axis=(0, 1)) + 1e-8)).astype(np.float32)
        v_scale = np.clip(v_scale, 0.25, 4.0)

        col_start = group_idx * d
        col_end = col_start + d
        refined_P_k[:, col_start:col_end] *= k_scale.reshape(1, d)
        refined_P_v[:, col_start:col_end] *= v_scale.reshape(1, d)
        diag_info.append({
            "group": group_idx,
            "k_scale_mean": float(np.mean(k_scale)),
            "k_scale_std": float(np.std(k_scale)),
            "v_scale_mean": float(np.mean(v_scale)),
            "v_scale_std": float(np.std(v_scale)),
        })

    return refined_P_k, refined_P_v, diag_info


def attention_proxy_error(Q_act, K_act, V_act, P_q, premix, P_k, P_v, mode="full_vha"):
    """Approximate attention behavior using per-token headwise logits/value proxies."""
    N = Q_act.shape[0]
    q_per_kv = SRC_H_Q // SRC_H_K
    target_q_heads = TGT_H_Q if mode == "full_vha" else KV_ONLY_H_Q

    if _gpu_available():
        Q_heads_t = _to_gpu(Q_act).reshape([N, SRC_H_Q, d])
        K_heads_t = _to_gpu(K_act).reshape([N, SRC_H_K, d])
        V_target_t = _to_gpu(build_virtual_v_targets(V_act))
        Q_base_t = paddle.matmul(_to_gpu(Q_act), _to_gpu(P_q)).reshape([N, target_q_heads, d])
        K_new_t = paddle.matmul(_to_gpu(K_act), _to_gpu(P_k)).reshape([N, TGT_H_K, d])
        V_new_t = paddle.matmul(_to_gpu(V_act), _to_gpu(P_v)).reshape([N, TGT_H_K, d])
        premix_t = _to_gpu(premix)
        logit_sse_t = paddle.zeros([], dtype="float32")
        logit_ref_t = paddle.zeros([], dtype="float32")
        out_sse_t = paddle.zeros([], dtype="float32")
        out_ref_t = paddle.zeros([], dtype="float32")
        for group_idx in range(TGT_H_K):
            if mode == "full_vha":
                Q_group_t = paddle.matmul(Q_base_t, premix_t[group_idx])
                q_start = group_idx * TGT_H_Q
                q_count = TGT_H_Q
            else:
                q_count = KV_ONLY_H_Q // TGT_H_K
                q_start = group_idx * q_count
                Q_group_t = Q_base_t[:, q_start:q_start + q_count, :]
            K_group_t = K_new_t[:, group_idx, :]
            V_group_t = V_new_t[:, group_idx, :]
            for q_idx in range(q_count):
                virtual_head = q_start + q_idx
                orig_kv = virtual_head // q_per_kv
                ref_logit_t = paddle.sum(Q_heads_t[:, virtual_head, :] * K_heads_t[:, orig_kv, :], axis=-1)
                pred_logit_t = paddle.sum(Q_group_t[:, q_idx, :] * K_group_t, axis=-1)
                logit_sse_t += paddle.sum((pred_logit_t - ref_logit_t) ** 2)
                logit_ref_t += paddle.sum(ref_logit_t ** 2)
                ref_out_t = V_target_t[:, virtual_head, :]
                out_sse_t += paddle.sum((V_group_t - ref_out_t) ** 2)
                out_ref_t += paddle.sum(ref_out_t ** 2)
        return {
            "proxy_logit_rel_rmse": float(paddle.sqrt(logit_sse_t / (logit_ref_t + 1e-8)).numpy()),
            "proxy_value_rel_rmse": float(paddle.sqrt(out_sse_t / (out_ref_t + 1e-8)).numpy()),
        }

    Q_heads = Q_act.reshape(N, SRC_H_Q, d)
    K_heads = K_act.reshape(N, SRC_H_K, d)
    V_target = build_virtual_v_targets(V_act)
    Q_base = (Q_act @ P_q).reshape(N, target_q_heads, d)
    K_new = (K_act @ P_k).reshape(N, TGT_H_K, d)
    V_new = (V_act @ P_v).reshape(N, TGT_H_K, d)
    logit_sse = 0.0
    logit_ref = 0.0
    out_sse = 0.0
    out_ref = 0.0
    for group_idx in range(TGT_H_K):
        if mode == "full_vha":
            Q_group = np.einsum("nqd,de->nqe", Q_base, premix[group_idx])
            q_start = group_idx * TGT_H_Q
            q_count = TGT_H_Q
        else:
            q_count = KV_ONLY_H_Q // TGT_H_K
            q_start = group_idx * q_count
            Q_group = Q_base[:, q_start:q_start + q_count, :]
        K_group = K_new[:, group_idx, :]
        V_group = V_new[:, group_idx, :]
        for q_idx in range(q_count):
            virtual_head = q_start + q_idx
            orig_kv = virtual_head // q_per_kv
            ref_logit = np.sum(Q_heads[:, virtual_head, :] * K_heads[:, orig_kv, :], axis=-1)
            pred_logit = np.sum(Q_group[:, q_idx, :] * K_group, axis=-1)
            logit_sse += float(np.sum((pred_logit - ref_logit) ** 2))
            logit_ref += float(np.sum(ref_logit ** 2))
            pred_out = V_group
            ref_out = V_target[:, virtual_head, :]
            out_sse += float(np.sum((pred_out - ref_out) ** 2))
            out_ref += float(np.sum(ref_out ** 2))
    return {
        "proxy_logit_rel_rmse": float((logit_sse / (logit_ref + 1e-8)) ** 0.5),
        "proxy_value_rel_rmse": float((out_sse / (out_ref + 1e-8)) ** 0.5),
    }

def optimize_q_premix(Q_act, num_iters=JOINT_OPT_ITERS):
    """
    Jointly optimize Q_base and premix via alternating least-squares.
    
    Goal: find Q_base[N,8,128] and premix[2,128,128] such that
        Q_base @ premix[k] ≈ Q_orig_group[k]  for k=0,1
    
    where Q_orig_group[0] = Q_act[:, 0:8, :] and Q_orig_group[1] = Q_act[:, 8:16, :]
    
    Args:
        Q_act: [N, 16*128=2048] - Q activations
    
    Returns:
        P_q: [2048, 1024] - projection from original Q space to Q_base space
        premix: [2, 128, 128]
    """
    N = Q_act.shape[0]
    Q_heads = Q_act.reshape(N, SRC_H_Q, d)  # [N, 16, 128]
    
    # Target groups
    Q_group0 = Q_heads[:, :TGT_H_Q, :]   # [N, 8, 128] - first 8 heads
    Q_group1 = Q_heads[:, TGT_H_Q:, :]   # [N, 8, 128] - last 8 heads
    Q_groups = [Q_group0, Q_group1]
    
    Q_base = init_q_base_from_groups(Q_heads)
    
    premix = np.zeros((TGT_H_K, d, d), dtype=np.float32)
    
    for iteration in range(num_iters):
        # === Step A: Fix Q_base, solve premix[k] ===
        for k in range(TGT_H_K):
            # Q_base @ premix[k] ≈ Q_groups[k]
            A = Q_base.reshape(-1, d)          # [N*8, 128]
            B = Q_groups[k].reshape(-1, d)     # [N*8, 128]
            
            ATA = A.T @ A + 1e-6 * np.eye(d, dtype=np.float32)
            ATB = A.T @ B
            premix[k] = np.linalg.solve(ATA, ATB)  # [128, 128]
        
        # === Step B: Fix premix, solve Q_base ===
        # For each head q: Q_base[:,q,:] @ premix[0] ≈ Q_group0[:,q,:]
        #                   Q_base[:,q,:] @ premix[1] ≈ Q_group1[:,q,:]
        # Stack: Q_base[q] @ [premix[0] | premix[1]] ≈ [Q_group0[q] | Q_group1[q]]
        
        M = np.concatenate([premix[0], premix[1]], axis=1)  # [128, 256]
        MMT = M @ M.T + 1e-6 * np.eye(d, dtype=np.float32)  # [128, 128]
        MMT_inv = np.linalg.inv(MMT)
        
        Q_base_new = np.zeros_like(Q_base)
        for q in range(TGT_H_Q):
            T = np.concatenate([Q_groups[0][:, q, :], Q_groups[1][:, q, :]], axis=1)  # [N, 256]
            # Q_base[q] = T @ M^T @ (M @ M^T)^{-1}
            Q_base_new[:, q, :] = T @ M.T @ MMT_inv  # [N, 128]
        
        Q_base = Q_base_new
        
        # Compute error
        total_err = 0.0
        for k in range(TGT_H_K):
            recon = np.einsum('nqd,de->nqe', Q_base, premix[k])
            err = np.linalg.norm(recon - Q_groups[k]) / np.linalg.norm(Q_groups[k])
            total_err += err
        avg_err = total_err / TGT_H_K
        
        if iteration == 0 or iteration == num_iters - 1:
            print(f"    Iter {iteration}: Q+premix avg error = {avg_err:.4f}")
    
    # Recover P_q_final: projection from original Q space to optimized Q_base
    # Q_base_flat = Q_act @ P_q_final
    Q_base_flat = Q_base.reshape(N, TGT_H_Q * d)  # [N, 1024]
    QTQ = Q_act.T @ Q_act + 1e-6 * np.eye(SRC_H_Q * d, dtype=np.float32)
    QTB = Q_act.T @ Q_base_flat
    P_q_final = np.linalg.solve(QTQ, QTB)  # [2048, 1024]
    
    return P_q_final, premix


def compute_vha_pre_o_proxy(Q_act, K_act, V_act, P_q, premix, P_k, P_v, seq_length, mode="full_vha", max_sequences=None):
    """Recompute a causal pre-O head-output proxy for the converted VHA layer."""
    N = Q_act.shape[0]
    usable_tokens = (N // seq_length) * seq_length
    if max_sequences is not None:
        usable_tokens = min(usable_tokens, max_sequences * seq_length)
    if usable_tokens == 0:
        return None

    target_q_heads = TGT_H_Q if mode == "full_vha" else KV_ONLY_H_Q
    num_sequences = usable_tokens // seq_length
    scale = 1.0 / np.sqrt(d)

    if _gpu_available():
        Q_base_t = paddle.matmul(_to_gpu(Q_act[:usable_tokens]), _to_gpu(P_q)).reshape([num_sequences, seq_length, target_q_heads, d])
        K_new_t = paddle.matmul(_to_gpu(K_act[:usable_tokens]), _to_gpu(P_k)).reshape([num_sequences, seq_length, TGT_H_K, d])
        V_new_t = paddle.matmul(_to_gpu(V_act[:usable_tokens]), _to_gpu(P_v)).reshape([num_sequences, seq_length, TGT_H_K, d])
        outputs_t = paddle.zeros([num_sequences, seq_length, target_q_heads, d], dtype="float32")
        causal_mask_t = paddle.to_tensor(np.triu(np.ones((seq_length, seq_length), dtype=bool), k=1), place=paddle.CUDAPlace(0))
        premix_t = _to_gpu(premix)
        for seq_idx in range(num_sequences):
            if mode == "full_vha":
                for group_idx in range(TGT_H_K):
                    Q_group_t = paddle.matmul(Q_base_t[seq_idx], premix_t[group_idx])
                    K_group_t = K_new_t[seq_idx, :, group_idx, :]
                    V_group_t = V_new_t[seq_idx, :, group_idx, :]
                    logits_t = paddle.einsum("thd,sd->hts", Q_group_t, K_group_t) * scale
                    logits_t = paddle.where(causal_mask_t.unsqueeze(0), paddle.full_like(logits_t, -1e9), logits_t)
                    probs_t = paddle.nn.functional.softmax(logits_t, axis=-1)
                    group_out_t = paddle.einsum("hts,sd->thd", probs_t, V_group_t)
                    outputs_t[seq_idx, :, group_idx * TGT_H_Q:(group_idx + 1) * TGT_H_Q, :] = group_out_t
            else:
                q_heads_per_kv = KV_ONLY_H_Q // TGT_H_K
                for group_idx in range(TGT_H_K):
                    q_start = group_idx * q_heads_per_kv
                    q_end = q_start + q_heads_per_kv
                    Q_group_t = Q_base_t[seq_idx, :, q_start:q_end, :]
                    K_group_t = K_new_t[seq_idx, :, group_idx, :]
                    V_group_t = V_new_t[seq_idx, :, group_idx, :]
                    logits_t = paddle.einsum("thd,sd->hts", Q_group_t, K_group_t) * scale
                    logits_t = paddle.where(causal_mask_t.unsqueeze(0), paddle.full_like(logits_t, -1e9), logits_t)
                    probs_t = paddle.nn.functional.softmax(logits_t, axis=-1)
                    outputs_t[seq_idx, :, q_start:q_end, :] = paddle.einsum("hts,sd->thd", probs_t, V_group_t)
        return _to_numpy(outputs_t.reshape([usable_tokens, target_q_heads * d]))

    Q_base = (Q_act[:usable_tokens] @ P_q).reshape(-1, seq_length, target_q_heads, d)
    K_new = (K_act[:usable_tokens] @ P_k).reshape(-1, seq_length, TGT_H_K, d)
    V_new = (V_act[:usable_tokens] @ P_v).reshape(-1, seq_length, TGT_H_K, d)
    outputs = np.zeros((num_sequences, seq_length, target_q_heads, d), dtype=np.float32)
    causal_mask = np.triu(np.ones((seq_length, seq_length), dtype=bool), k=1)

    for seq_idx in range(num_sequences):
        if mode == "full_vha":
            for group_idx in range(TGT_H_K):
                Q_group = Q_base[seq_idx] @ premix[group_idx]
                K_group = K_new[seq_idx, :, group_idx, :]
                V_group = V_new[seq_idx, :, group_idx, :]
                logits = np.einsum("thd,sd->hts", Q_group, K_group) * scale
                logits[:, causal_mask] = -1e9
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                probs = np.exp(logits).astype(np.float32)
                probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-8)
                group_out = np.einsum("hts,sd->thd", probs, V_group)
                outputs[seq_idx, :, group_idx * TGT_H_Q:(group_idx + 1) * TGT_H_Q, :] = group_out
        else:
            q_heads_per_kv = KV_ONLY_H_Q // TGT_H_K
            for group_idx in range(TGT_H_K):
                q_start = group_idx * q_heads_per_kv
                q_end = q_start + q_heads_per_kv
                Q_group = Q_base[seq_idx, :, q_start:q_end, :]
                K_group = K_new[seq_idx, :, group_idx, :]
                V_group = V_new[seq_idx, :, group_idx, :]
                logits = np.einsum("thd,sd->hts", Q_group, K_group) * scale
                logits[:, causal_mask] = -1e9
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                probs = np.exp(logits).astype(np.float32)
                probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-8)
                outputs[seq_idx, :, q_start:q_end, :] = np.einsum("hts,sd->thd", probs, V_group)
    return outputs.reshape(usable_tokens, target_q_heads * d)

# =============================================================================
# Step 3: Postmix initialization
# =============================================================================

def build_virtual_v_targets(V_act):
    """Map original GQA V heads to VHA virtual heads."""
    N = V_act.shape[0]
    V_heads = V_act.reshape(N, SRC_H_K, d)
    q_per_kv = SRC_H_Q // SRC_H_K
    targets = np.zeros((N, SRC_H_Q, d), dtype=np.float32)
    for virtual_head in range(SRC_H_Q):
        orig_v_idx = virtual_head // q_per_kv
        targets[:, virtual_head, :] = V_heads[:, orig_v_idx, :]
    return targets


def init_postmix_pre_o(pre_o_ref, pre_o_base, rank=POSTMIX_RANK, num_heads=SRC_H_Q):
    """Initialize postmix by fitting the real pre-O head output mixing objective."""
    if pre_o_ref is None or pre_o_base is None:
        return zero_postmix(rank=rank)

    X_heads = pre_o_base.reshape(pre_o_base.shape[0], num_heads, d)
    Y_heads = pre_o_ref.reshape(pre_o_ref.shape[0], num_heads, d)
    X = X_heads.transpose(0, 2, 1).reshape(-1, num_heads)
    Y = Y_heads.transpose(0, 2, 1).reshape(-1, num_heads)

    if _gpu_available():
        X_t = _to_gpu(X)
        Y_t = _to_gpu(Y)
        eye_t = paddle.eye(num_heads, dtype="float32")
        XtX = paddle.matmul(X_t, X_t, transpose_x=True) + 1e-3 * eye_t
        XtY = paddle.matmul(X_t, Y_t, transpose_x=True)
        A_t = paddle.linalg.lstsq(XtX, XtY, rcond=1e-5)[0]
        residual_mix_t = A_t - eye_t
        U_svd, S_svd, V_svd = paddle.linalg.svd(residual_mix_t, full_matrices=False)
        sqrt_S = paddle.sqrt(paddle.clip(S_svd[:rank], min=0.0) + 1e-8)
        postmix_U_init = U_svd[:, :rank] * sqrt_S
        postmix_V_init = V_svd[:, :rank] * sqrt_S
        mix_t = eye_t + paddle.matmul(postmix_U_init, postmix_V_init, transpose_y=True)
        pred_after = paddle.matmul(X_t, mix_t)
        before_err = paddle.linalg.norm(X_t - Y_t) / (paddle.linalg.norm(Y_t) + 1e-8)
        after_err = paddle.linalg.norm(pred_after - Y_t) / (paddle.linalg.norm(Y_t) + 1e-8)
        before_value = float(before_err.numpy())
        after_value = float(after_err.numpy())
        if after_value >= before_value:
            U_zero, V_zero, _, _ = zero_postmix(rank=rank, num_heads=num_heads)
            return U_zero, V_zero, before_value, before_value
        return _to_numpy(postmix_U_init), _to_numpy(postmix_V_init), before_value, after_value

    XtX = X.T @ X + 1e-3 * np.eye(num_heads, dtype=np.float32)
    XtY = X.T @ Y
    A = np.linalg.lstsq(XtX, XtY, rcond=1e-5)[0].astype(np.float32)
    residual_mix = A - np.eye(num_heads, dtype=np.float32)

    U_svd, S_svd, Vt_svd = np.linalg.svd(residual_mix, full_matrices=False)
    sqrt_S = np.sqrt(np.maximum(S_svd[:rank], 0.0) + 1e-8)
    postmix_U_init = (U_svd[:, :rank] * sqrt_S).astype(np.float32)
    postmix_V_init = (Vt_svd[:rank, :].T * sqrt_S).astype(np.float32)

    pred_after = X @ (np.eye(num_heads, dtype=np.float32) + postmix_U_init @ postmix_V_init.T)
    before_err = float(np.linalg.norm(X - Y) / (np.linalg.norm(Y) + 1e-8))
    after_err = float(np.linalg.norm(pred_after - Y) / (np.linalg.norm(Y) + 1e-8))
    if after_err >= before_err:
        U_zero, V_zero, _, _ = zero_postmix(rank=rank, num_heads=num_heads)
        return U_zero, V_zero, before_err, before_err
    return postmix_U_init, postmix_V_init, before_err, after_err


def zero_postmix(rank=POSTMIX_RANK, num_heads=SRC_H_Q):
    return (
        np.zeros((num_heads, rank), dtype=np.float32),
        np.zeros((num_heads, rank), dtype=np.float32),
        0.0,
        0.0,
    )


def init_postmix_v_proxy(V_act, P_v, rank=POSTMIX_RANK):
    """Fallback postmix proxy based only on V projection activations."""
    N = V_act.shape[0]
    V_compressed = V_act @ P_v
    V_new_heads = V_compressed.reshape(N, TGT_H_K, d)

    base = np.zeros((N, SRC_H_Q, d), dtype=np.float32)
    for group_idx in range(TGT_H_K):
        base[:, group_idx * TGT_H_Q:(group_idx + 1) * TGT_H_Q, :] = V_new_heads[:, group_idx:group_idx + 1, :]

    target = build_virtual_v_targets(V_act)
    return init_postmix_pre_o(target.reshape(N, SRC_H_Q * d), base.reshape(N, SRC_H_Q * d), rank=rank)

# =============================================================================
# Step 4: Assemble VHA checkpoint
# =============================================================================


def save_conversion_part(part_dir, layer_idx, P_k, P_v, P_q, premix, U_post, V_post, layer_diag):
    os.makedirs(part_dir, exist_ok=True)
    part_path = os.path.join(part_dir, f"layer{layer_idx}.pkl")
    with open(part_path, "wb") as f:
        pickle.dump({
            "P_k": P_k,
            "P_v": P_v,
            "P_q": P_q,
            "premix": premix,
            "postmix_U": U_post,
            "postmix_V": V_post,
            "diagnostics": layer_diag,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved conversion part: {part_path}", flush=True)


def load_conversion_parts(part_dir):
    all_P_k, all_P_v, all_P_q, all_premix = {}, {}, {}, {}
    all_postmix_U, all_postmix_V = {}, {}
    diagnostics_layers = []
    for layer_idx in range(NUM_LAYERS):
        part_path = os.path.join(part_dir, f"layer{layer_idx}.pkl")
        if not os.path.exists(part_path):
            raise FileNotFoundError(f"Missing conversion part: {part_path}")
        with open(part_path, "rb") as f:
            part = pickle.load(f)
        all_P_k[layer_idx] = part["P_k"]
        all_P_v[layer_idx] = part["P_v"]
        all_P_q[layer_idx] = part["P_q"]
        all_premix[layer_idx] = part["premix"]
        all_postmix_U[layer_idx] = part["postmix_U"]
        all_postmix_V[layer_idx] = part["postmix_V"]
        diagnostics_layers.append(part["diagnostics"])
    return all_P_k, all_P_v, all_P_q, all_premix, all_postmix_U, all_postmix_V, diagnostics_layers

def assemble_vha_checkpoint(layers, other_tensors, all_P_k, all_P_v, all_P_q, all_premix, all_postmix_U, all_postmix_V, mode="full_vha"):
    """Assemble final VHA checkpoint tensors."""
    import ml_dtypes
    
    output_tensors = {}
    
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}.self_attn"
        
        W_q = layers[layer_idx]["W_q"]  # [D, 2048]
        W_k = layers[layer_idx]["W_k"]  # [D, 1024]
        W_v = layers[layer_idx]["W_v"]  # [D, 1024]
        W_o = layers[layer_idx]["W_o"]  # [2048, D]
        
        P_k = all_P_k[layer_idx]    # [1024, 256]
        P_v = all_P_v[layer_idx]    # [1024, 256]
        P_q = all_P_q[layer_idx]
        premix = all_premix[layer_idx]
        
        # New weights
        W_q_new = W_q @ P_q if mode == "full_vha" else W_q
        W_k_new = W_k @ P_k
        W_v_new = W_v @ P_v
        target_q_heads = TGT_H_Q if mode == "full_vha" else KV_ONLY_H_Q
        
        # Fused qkv in PaddleFleet grouped layout:
        # [G0_Q, G0_K, G0_V, G1_Q, G1_K, G1_V]
        qkv_fused = pack_grouped_qkv_lastdim(
            W_q_new, W_k_new, W_v_new, target_q_heads, TGT_H_K, d
        )
        
        output_tensors[f"{prefix}.qkv_proj.weight"] = qkv_fused.astype(ml_dtypes.bfloat16)
        output_tensors[f"{prefix}.o_proj.weight"] = W_o.astype(ml_dtypes.bfloat16)
        if mode == "full_vha":
            output_tensors[f"{prefix}.vha_premix_weight"] = premix.astype(ml_dtypes.bfloat16)
        output_tensors[f"{prefix}.vha_postmix_U"] = all_postmix_U[layer_idx].astype(ml_dtypes.bfloat16)
        output_tensors[f"{prefix}.vha_postmix_V"] = all_postmix_V[layer_idx].astype(ml_dtypes.bfloat16)
        
        # Copy norms
        if layers[layer_idx]["q_norm"] is not None:
            output_tensors[f"{prefix}.q_norm.weight"] = layers[layer_idx]["q_norm"].astype(ml_dtypes.bfloat16)
        if layers[layer_idx]["k_norm"] is not None:
            output_tensors[f"{prefix}.k_norm.weight"] = layers[layer_idx]["k_norm"].astype(ml_dtypes.bfloat16)
    
    
    # Copy non-attention tensors (MLP, norms, embeddings)
    # Source already has up_gate_proj fused, just copy through
    for key, val in other_tensors.items():
        out_key = key if key.startswith("model.") else f"model.{key}"
        if "q_norm" in out_key or "k_norm" in out_key:
            continue
        output_tensors[out_key] = val.astype(ml_dtypes.bfloat16) if val.dtype == np.float32 else val

    return output_tensors


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Activation-based GQA -> VHA conversion")
    parser.add_argument("--gqa_checkpoint", type=str, required=True,
                        help="Path to GQA checkpoint directory (model_state_merged)")
    parser.add_argument("--gqa_model_config", type=str, required=True,
                        help="Path to GQA model config directory (contains config.json)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for VHA checkpoint")
    parser.add_argument("--calib_data", type=str, default=None,
                        help="Path to calibration data (mmap format). If None, uses random proxy.")
    parser.add_argument("--num_calib_samples", type=int, default=512)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--max_calib_tokens", type=int, default=200000,
                        help="Maximum activation tokens per layer used while collecting activations")
    parser.add_argument("--conversion_max_tokens", type=int, default=65536,
                        help="Maximum cached activation tokens per layer used for conversion")
    parser.add_argument("--joint_iters", type=int, default=5,
                        help="Alternating optimization iterations for Q+premix")
    parser.add_argument("--kv_blend_mean", type=float, default=0.1,
                        help="Conservative mean-basis blend used in final attention-aware groupwise KV fusion.")
    parser.add_argument("--activation_cache", type=str, default=None,
                        help="Path to reusable GQA activation cache (.npz).")
    parser.add_argument("--save_activation_cache", action="store_true",
                        help="Save collected GQA activations to --activation_cache for future runs.")
    parser.add_argument("--refresh_activation_cache", action="store_true",
                        help="Ignore existing --activation_cache and recollect activations.")
    parser.add_argument("--activation_shard_rank", type=int, default=0,
                        help="Rank of this activation collection shard.")
    parser.add_argument("--activation_shard_count", type=int, default=1,
                        help="Total number of activation collection shards.")
    parser.add_argument("--activation_batch_size", type=int, default=1,
                        help="Per-forward batch size used while collecting activations.")
    parser.add_argument("--conversion_layer_rank", type=int, default=0,
                        help="Layer shard rank for parallel conversion.")
    parser.add_argument("--conversion_layer_count", type=int, default=1,
                        help="Total layer shards for parallel conversion.")
    parser.add_argument("--conversion_part_dir", type=str, default=None,
                        help="Directory for per-layer conversion part files.")
    parser.add_argument("--assemble_from_parts", type=str, default=None,
                        help="Assemble final checkpoint from conversion part directory and skip per-layer conversion.")
    parser.add_argument("--collect_only", action="store_true",
                        help="Only collect/save activations; skip conversion and checkpoint assembly.")
    parser.add_argument("--conversion_mode", choices=["full_vha", "kv_postmix_only"], default="full_vha",
                        help="full_vha fits Q+premix; kv_postmix_only keeps 16 Q heads and disables premix.")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load GQA weights
    print("=" * 60)
    print("Step 0: Loading GQA checkpoint")
    print("=" * 60)
    layers, other_tensors = load_gqa_weights(args.gqa_checkpoint)
    print(f"  Loaded {NUM_LAYERS} layers, {len(other_tensors)} other tensors")
    
    # Collect or load reusable GQA activations
    print("\n" + "=" * 60)
    print("Step 0b: Preparing GQA activations")
    print("=" * 60)
    activations = None
    if args.activation_cache and os.path.exists(args.activation_cache) and not args.refresh_activation_cache:
        print(f"  Will lazily read merged activation cache: {args.activation_cache}", flush=True)
    else:
        activations = collect_activations_real(layers, other_tensors, args.calib_data, args.num_calib_samples, args.seq_length, args=args)
        if args.activation_cache and args.save_activation_cache:
            save_activation_cache(activations, args.activation_cache, {
                "gqa_checkpoint": args.gqa_checkpoint,
                "gqa_model_config": args.gqa_model_config,
                "calib_data": args.calib_data,
                "num_calib_samples": args.num_calib_samples,
                "seq_length": args.seq_length,
                "max_calib_tokens": args.max_calib_tokens,
                "activation_shard_rank": args.activation_shard_rank,
                "activation_shard_count": args.activation_shard_count,
                "contains": ["Q", "K", "V", "pre_o"],
            })
    if args.collect_only:
        print("  collect_only set; skipping conversion", flush=True)
        return
    
    # Per-layer conversion
    all_P_k, all_P_v, all_P_q = {}, {}, {}
    all_premix = {}
    all_postmix_U, all_postmix_V = {}, {}
    final_recipe = {
        "name": "vha_kv_postmix_activation_fusion",
        "kv": "DHA-style exhaustive balanced grouping with fixed attention-aware loss, then TransMLA-style balanced groupwise PCA with conservative mean blending",
        "q_premix": "current full VHA mode still fits Q base and premix; KV-only experiments should keep Q/premix fixed or bypass them in a separate target config",
        "joint_refine": "one-shot per-dimension K/V gain calibration against QK-logit and value proxy losses after Q+premix fitting",
        "postmix": "pre-O head-output low-rank fitting after attention-aware KV calibration; GQA activations are cacheable and reusable",
        "diagnostics": "per-layer K/V reconstruction, grouping loss decomposition, diagonal refine deltas, postmix residual, and cheap QK/V attention proxy errors",
        "recommended_refine": "run refine_vha_alignment.py first with --train_mode vha_o to align VHA/postmix and o_proj, then warmup with strict stop_gradient freezing",
        "loss_policy": "conversion optimizes continuous internal states with MSE/L2 and attention-aware QK/V proxies; final teacher behavior should be checked with logits KL/eval loss during refine/warmup",
    }
    part_dir = args.conversion_part_dir or os.path.join(args.output_path, "conversion_parts")
    diagnostics = {
        "recipe": final_recipe,
        "conversion_mode": args.conversion_mode,
        "kv_blend_mean": args.kv_blend_mean,
        "kv_grouping": "exhaustive_attention_aware_balanced_search",
        "q_init_method": "avg_then_als" if args.conversion_mode == "full_vha" else "identity_q_no_premix",
        "activation_cache": args.activation_cache,
        "activation_cache_used": bool(args.activation_cache and os.path.exists(args.activation_cache) and not args.refresh_activation_cache),
        "joint_diag_refine": True,
        "attention_proxy_diag": True,
        "loss_notes": {
            "fusion_mse": "Used for hidden/activation/head fusion because these are continuous internal states, matching DHA fusion loss style.",
            "attention_proxy": "Cheap per-token QK dot-product plus causal pre-O head-output/postmix diagnostics; still not a substitute for block/logit evaluation.",
            "logit_kl": "Use after checkpoint construction for teacher-student warmup/eval; KL is defined on output token distributions, not on raw K/V head activations.",
        },
        "layers": [],
    }
    
    print("\n" + "=" * 60)
    print("Step 1-3: Per-layer activation-based conversion")
    print("=" * 60)
    
    for layer_idx in ([] if args.assemble_from_parts else range(NUM_LAYERS)):
        if layer_idx % args.conversion_layer_count != args.conversion_layer_rank:
            continue
        print(f"\n--- Layer {layer_idx} ---", flush=True)
        
        if args.activation_cache and os.path.exists(args.activation_cache) and not args.refresh_activation_cache:
            layer_activations = load_activation_layer(args.activation_cache, layer_idx, max_tokens=args.conversion_max_tokens)
        else:
            layer_activations = activations[layer_idx]
        K_act = layer_activations["K"]
        V_act = layer_activations["V"]
        Q_act = layer_activations["Q"]
        pre_o_ref = layer_activations.get("pre_o")
        
        layer_diag = {"layer": layer_idx}

        # Step 1: attention-aware KV fusion/compression
        P_k, P_v, k_err, v_err, group_errors = compress_kv_groupwise_pca(
            K_act, V_act, Q_act=Q_act, blend_mean=args.kv_blend_mean
        )
        layer_diag["group_errors"] = group_errors
        layer_diag["k_error"] = k_err
        layer_diag["v_error"] = v_err
        print(f"  Attention-aware KV PCA: K_error={k_err:.4f}, V_error={v_err:.4f}")
        for group_error in group_errors:
            if "search_summary" in group_error:
                print(f"    search_summary: loss={group_error['search_summary']['search_loss']:.6f}")
            else:
                print("    group {group}: K_heads={k_src_heads}, V_heads={v_src_heads}, K={k_error:.4f}, V={v_error:.4f}, scale={scale_k:.4f}".format(**group_error))
        
        # Step 2: Q + premix handling
        if args.conversion_mode == "full_vha":
            P_q, premix = optimize_q_premix(Q_act, num_iters=args.joint_iters)
        else:
            P_q = np.eye(SRC_H_Q * d, dtype=np.float32)
            premix = np.zeros((TGT_H_K, d, d), dtype=np.float32)

        before_refine_diag = attention_proxy_error(Q_act, K_act, V_act, P_q, premix, P_k, P_v, mode=args.conversion_mode)
        if args.conversion_mode == "full_vha":
            P_k, P_v, diag_refine = refine_kv_diagonal_attention(Q_act, K_act, V_act, P_q, premix, P_k, P_v)
        else:
            diag_refine = []
        after_refine_diag = attention_proxy_error(Q_act, K_act, V_act, P_q, premix, P_k, P_v, mode=args.conversion_mode)
        layer_diag["diag_refine"] = diag_refine
        layer_diag["proxy_before_diag_refine"] = before_refine_diag
        layer_diag["proxy_after_diag_refine"] = after_refine_diag
        k_err_after, v_err_after = compute_kv_reconstruction_error(K_act, V_act, P_k, P_v)
        layer_diag["k_error_after_diag_refine"] = k_err_after
        layer_diag["v_error_after_diag_refine"] = v_err_after
        layer_diag.update(after_refine_diag)
        print(
            "  Joint diag refine: QK {proxy_logit_rel_rmse:.4f}->{after_qk:.4f}, V {proxy_value_rel_rmse:.4f}->{after_v:.4f}".format(
                after_qk=after_refine_diag["proxy_logit_rel_rmse"],
                after_v=after_refine_diag["proxy_value_rel_rmse"],
                **before_refine_diag,
            )
        )

        all_P_k[layer_idx] = P_k
        all_P_v[layer_idx] = P_v
        all_P_q[layer_idx] = P_q
        all_premix[layer_idx] = premix
        
        # Step 3: Postmix initialization against pre-O head outputs
        pre_o_base = compute_vha_pre_o_proxy(Q_act, K_act, V_act, P_q, premix, P_k, P_v, args.seq_length, mode=args.conversion_mode)
        postmix_heads = TGT_H_Q if args.conversion_mode == "full_vha" else KV_ONLY_H_Q
        if pre_o_ref is not None and pre_o_base is not None:
            pre_o_ref_fit = pre_o_ref[:pre_o_base.shape[0]]
            expected_ref_width = postmix_heads * d
            if pre_o_ref_fit.shape[1] != expected_ref_width:
                # Full-VHA premix reduces query/head output width before o_proj; fit postmix in that reduced space.
                W_o = layers[layer_idx].get("W_o")
                if W_o is not None and W_o.shape[0] >= expected_ref_width:
                    pre_o_ref_fit = pre_o_ref_fit[:, :expected_ref_width]
                    postmix_target = "pre_o_head_output_reduced"
                else:
                    pre_o_ref_fit = None
                    postmix_target = "v_projection_fallback"
            else:
                postmix_target = "pre_o_head_output"
        else:
            pre_o_ref_fit = None
            postmix_target = "v_projection_fallback"
        if pre_o_ref_fit is not None:
            U_post, V_post, postmix_before, postmix_after = init_postmix_pre_o(pre_o_ref_fit, pre_o_base, num_heads=postmix_heads)
        else:
            U_post, V_post, postmix_before, postmix_after = init_postmix_v_proxy(V_act, P_v)
        postmix_improved = postmix_after < postmix_before - 1e-6
        print(
            f"  Postmix {postmix_target}: before={postmix_before:.4f}, "
            f"after={postmix_after:.4f}, improved={postmix_improved}"
        )
        layer_diag["postmix_target"] = postmix_target
        layer_diag["postmix_before_error"] = postmix_before
        layer_diag["postmix_after_error"] = postmix_after
        layer_diag["postmix_improved"] = bool(postmix_improved)
        all_postmix_U[layer_idx] = U_post
        all_postmix_V[layer_idx] = V_post
        diagnostics["layers"].append(layer_diag)
        save_conversion_part(part_dir, layer_idx, P_k, P_v, P_q, premix, U_post, V_post, layer_diag)
        
        # Free memory
        del layer_activations, K_act, V_act, Q_act
        if pre_o_ref is not None:
            del pre_o_ref
        if activations is not None:
            del activations[layer_idx]
        gc.collect()
    
    if args.assemble_from_parts:
        print(f"  Loading conversion parts from: {args.assemble_from_parts}", flush=True)
        (
            all_P_k,
            all_P_v,
            all_P_q,
            all_premix,
            all_postmix_U,
            all_postmix_V,
            diagnostics["layers"],
        ) = load_conversion_parts(args.assemble_from_parts)
    else:
        print(
            f"  Layer shard {args.conversion_layer_rank}/{args.conversion_layer_count} done; "
            f"parts saved to {part_dir}",
            flush=True,
        )
        return

    # Step 4: Assemble checkpoint
    print("\n" + "=" * 60)
    print("Step 4: Assembling VHA checkpoint")
    print("=" * 60)
    
    output_tensors = assemble_vha_checkpoint(
        layers, other_tensors,
        all_P_k, all_P_v, all_P_q, all_premix,
        all_postmix_U, all_postmix_V,
        mode=args.conversion_mode,
    )
    
    print(f"  Total tensors: {len(output_tensors)}")
    
    # Save
    output_file = os.path.join(args.output_path, "model-00001-of-00001.safetensors")
    print(f"  Saving to: {output_file}")
    save_file(output_tensors, output_file)
    
    # Save weight map index
    weight_map = {k: "model-00001-of-00001.safetensors" for k in output_tensors.keys()}
    total_size = sum(t.nbytes for t in output_tensors.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=4)
    
    # Save VHA config
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": D,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": TGT_H_Q if args.conversion_mode == "full_vha" else KV_ONLY_H_Q,
        "num_key_value_heads": TGT_H_K,
        "head_dim": d,
        "intermediate_size": 6144,
        "vocab_size": 151936,
        "attn_type": "vha",
        "vha_enable_premix": args.conversion_mode == "full_vha",
        "vha_enable_postmix": True,
        "vha_postmix_rank": POSTMIX_RANK,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
        "use_qk_norm": True,
    }
    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.output_path, "conversion_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    with open(os.path.join(args.output_path, "conversion_recipe.json"), "w") as f:
        json.dump(final_recipe, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
