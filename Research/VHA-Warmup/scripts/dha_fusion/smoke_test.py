#!/usr/bin/env python3
"""Smoke test for DHA fusion: load GQA, install hooks, single fwd+bwd."""
import os, sys, contextlib
import numpy as np
import paddle

os.environ.setdefault("FLAGS_enable_pir_api", "0")

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

from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
def _simple_linear(input_, weight, bias=None):
    out_shape = list(input_.shape[:-1]) + [weight.shape[-1]]
    flat = input_.reshape([-1, input_.shape[-1]]).cast("float32")
    out = paddle.matmul(flat, weight.cast("float32")).cast(input_.dtype).reshape(out_shape)
    if bias is not None:
        out = out + bias
    return out

def _simple_col_fwd(self, input_, weight=None, runtime_gather_output=None):
    w = weight if weight is not None else self.weight
    bias = self.bias if not self.skip_bias_add and self.bias is not None else None
    out = _simple_linear(input_, w, bias)
    ob = self.bias if self.skip_bias_add and self.bias is not None else None
    return out, ob

def _simple_row_fwd(self, input_, weight=None):
    w = weight if weight is not None else self.weight
    bias = self.bias if not self.skip_bias_add and self.bias is not None else None
    out = _simple_linear(input_, w, bias)
    ob = self.bias if self.skip_bias_add and self.bias is not None else None
    return out, ob

ColumnParallelLinear.forward = _simple_col_fwd
RowParallelLinear.forward = _simple_row_fwd

DHA = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/scripts/dha_fusion"
VHA = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA"
sys.path.insert(0, DHA)
sys.path.insert(0, VHA)

from models.qwen_provider import create_provider
from paddlefleet.transformer.transformer_layer import TransformerLayer

from grouping import load_groupings, validate_groupings
from attention_patch import install_all_fusion_hooks, reset_caches
from alm_loss import ALMConfig, ALMState, compute_total_constraint, alm_combine

GQA_CKPT = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged"
GQA_CONFIG = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA"
GROUPINGS = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_kv_postmix_activation_128/conversion_diagnostics.json"
SEQ_LEN, BATCH = 512, 2

print("[1] build GQA model from", GQA_CONFIG)
provider = create_provider(GQA_CONFIG)
provider.seq_length = SEQ_LEN
provider.max_sequence_length = SEQ_LEN
model = provider.provide()

layers = [fn for fn in model.run_function if isinstance(fn, TransformerLayer)]
print(f"  num_layers={len(layers)}")
attn0 = layers[0].self_attn
print(f"  attn type: {type(attn0).__name__}, core_attention: {type(attn0.core_attention).__name__}")

print("[2] load ckpt")
from safetensors import safe_open
param_dict = dict(model.named_parameters())

def ckpt_key_to_pipeline_name(key):
    k = key.replace("model.", "", 1) if key.startswith("model.") else key
    if k.startswith("embedding."):
        return f"0.{k}"
    if k.startswith("layers."):
        parts = k.split(".", 2)
        return f"{int(parts[1]) + 1}.{parts[2]}"
    if k == "norm.weight":
        return "29.norm.weight"
    if k == "lm_head.weight":
        return "30.weight"
    return None

sf_files = sorted(f for f in os.listdir(GQA_CKPT) if f.endswith(".safetensors"))
loaded, mismatch = 0, []
for sf in sf_files:
    with safe_open(os.path.join(GQA_CKPT, sf), framework="pt", device="cpu") as f:
        for key in f.keys():
            val = f.get_tensor(key).float().numpy()
            pname = ckpt_key_to_pipeline_name(key)
            p = param_dict.get(pname) if pname else None
            if p is None:
                for cand in [key, "model." + key, key.replace("model.", "")]:
                    p = param_dict.get(cand)
                    if p is not None: break
            if p is not None:
                if list(p.shape) == list(val.shape):
                    p.set_value(paddle.to_tensor(val).cast(p.dtype))
                    loaded += 1
                else:
                    mismatch.append((key, val.shape, list(p.shape)))
print(f"  loaded {loaded}/{len(param_dict)} params, mismatches={len(mismatch)}")
if mismatch:
    for k, vs, ps in mismatch[:5]: print(f"    MISMATCH {k}: ckpt={vs} model={ps}")
    raise SystemExit(1)

print("[3] install hooks")
groupings = load_groupings(GROUPINGS)
validate_groupings(groupings, n_layers=len(layers), n_src_heads=8, n_groups=2)
fusion_states = install_all_fusion_hooks(layers, groupings)
print(f"  installed {len(fusion_states)} hooks")

print(f"[4] forward B={BATCH} T={SEQ_LEN}")
input_ids = paddle.randint(100, 100000, (BATCH, SEQ_LEN), dtype="int64")
reset_caches(fusion_states)
x = {"input_ids": input_ids}
for fn in model.run_function:
    x = fn(x)
if isinstance(x, dict):
    print(f"  final keys: {list(x.keys())}")
    logits = x.get("logits", x.get("output"))
else:
    logits = x
print(f"  logits shape: {logits.shape}")

print("[5] verify caches")
for i, fs in enumerate(fusion_states[:3]):
    k = fs.cached_k_pre_fusion
    print(f"  layer {i}: K shape={k.shape if k is not None else None}")
    assert k is not None, f"layer {i} cache empty"

print("[6] losses")
cfg = ALMConfig(target_initial=-1.0, target_decay_steps=1500)
state = ALMState(cfg)
sh_logits = logits[:, :-1, :].cast("float32")
sh_labels = input_ids[:, 1:]
lm = paddle.nn.functional.cross_entropy(
    sh_logits.reshape([-1, sh_logits.shape[-1]]),
    sh_labels.reshape([-1]), reduction="mean")
constraint = compute_total_constraint(fusion_states)
total = alm_combine(lm, constraint, state)
print(f"  lm={float(lm.item()):.4f} C={float(constraint.item()):.4f} total={float(total.item()):.4f}")

print("[7] backward")
total.backward()
pd = dict(model.named_parameters())
for name in sorted(pd.keys()):
    if "1.self_attn" in name and ("linear_qkv" in name or "dha_fusion" in name):
        g = pd[name].grad
        gn = float(paddle.norm(g.cast("float32")).item()) if g is not None else None
        print(f"  {name}: grad_norm={gn}")

state.end_step(float(constraint.item()), float(lm.item()))
print(f"  state.target_initial after end_step: {state.target_initial:.4f}")
print("SMOKE TEST PASSED")
