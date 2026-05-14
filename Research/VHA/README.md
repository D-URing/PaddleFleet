# VHA: Virtual Head Attention

PaddleFleet-based pre-training framework for VHA and GQA baseline experiments.

## Project Structure

```
Research/VHA/
├── run_pretrain.py              # Training entry point
├── models/                      # Model implementations
│   ├── vha_attention.py         # VHASelfAttention (extends PaddleFleet SelfAttention)
│   ├── vha_builder.py           # VHA layer spec + model builder
│   └── qwen_provider.py        # Dynamic config provider (GQA/VHA)
├── utils/                       # Utilities
│   └── warmup.py               # VHA warmup manager + GQA checkpoint conversion
├── config/                      # Configurations
│   └── qwen3/
│       ├── Qwen3-1.7B-GQA/config.json      # GQA model architecture
│       ├── Qwen3-1.7B-VHA/config.json      # VHA model architecture
│       ├── qwen3_gqa_1p7B_debug.json       # GQA debug (1 GPU)
│       ├── qwen3_gqa_1p7B_pretrain.json    # GQA pretrain (8 GPU)
│       ├── qwen3_vha_1p7B_debug.json       # VHA debug (1 GPU)
│       ├── qwen3_vha_1p7B_pretrain.json    # VHA pretrain (8 GPU)
│       └── qwen3_vha_1p7B_warmup.json      # VHA warmup from GQA ckpt
├── scripts/                     # Launch utilities
│   ├── train.sh                 # Generic training launcher
│   └── kill_process.sh          # Process cleanup
├── exps/                        # Experiment orchestration
│   └── pretrain.sh              # Experiment name -> config mapping
└── output/                      # Checkpoints and logs (gitignored)
```

## Quick Start

```bash
# Debug (single GPU, GQA baseline)
bash exps/pretrain.sh qwen3_gqa_1p7B_debug

# Debug (single GPU, VHA)
bash exps/pretrain.sh qwen3_vha_1p7B_debug

# Full pretrain (8 GPU, GQA baseline)
bash exps/pretrain.sh qwen3_gqa_1p7B_pretrain_8gpu

# Full pretrain (8 GPU, VHA from scratch)
bash exps/pretrain.sh qwen3_vha_1p7B_pretrain_8gpu

# VHA warmup from GQA checkpoint
bash exps/pretrain.sh qwen3_vha_1p7B_warmup_8gpu

# Or launch directly with config
bash scripts/train.sh config/qwen3/qwen3_vha_1p7B_pretrain.json 1 8
```

## VHA Architecture

VHA extends GQA with two lightweight mechanisms:

- **Premix**: Per-KV-group rotation matrices expand Q heads into virtual heads via learned rotations
- **Postmix**: Low-rank cross-head mixing `(I + VU^T)` applied after attention

Both are fully absorbable at inference — premix folds into W_q, postmix folds into W_o — so inference cost equals standard GQA.

## Training Modes

| Mode | Provider | Description |
|------|----------|-------------|
| GQA from scratch | `qwen_gqa_*` | Standard GQA baseline |
| VHA from scratch | `qwen_vha_*` | VHA with identity-init premix/postmix |
| VHA warmup | `qwen_vha_*` + `vha_warmup_from_gqa` | Load GQA ckpt, init VHA to identity, phased unfreeze |

## VHA Warmup Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `vha_warmup_from_gqa` | None | Path to GQA checkpoint directory |
| `vha_stabilize_steps` | 0 | Steps to keep premix/postmix frozen |
| `vha_premix_lr_scale` | 1.0 | LR multiplier for premix params |
| `vha_postmix_lr_scale` | 1.0 | LR multiplier for postmix params |

## Requirements

- PaddlePaddle >= 3.4 (nightly)
- PaddleFleet (this repo)
- PaddleFormers (in .deps/)
