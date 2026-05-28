# VHA Clean Pipeline

## 执行

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup
bash scripts/run_vha_pipeline.sh <stage>
```

常用一键命令：

```bash
bash scripts/run_vha_pipeline.sh all
```

如果需要从头清理：

```bash
bash scripts/run_vha_pipeline.sh clean
bash scripts/run_vha_pipeline.sh all
```

## 功能

本文档描述当前干净的 GQA → VHA 转换、评估、refine、warmup 后训练完整链路。新流程只服务于 `qwen3_vha_1p7B_clean_kv_postmix`，不混用旧 warmup001-015 实验产物。

## 目标结构

- GQA teacher：`16 Q heads`，`8 KV heads`。
- VHA student：`16 Q heads`，`2 KV heads`。
- 只做 KV 压缩，不做 Q 压缩。
- 启用 postmix，不启用 premix。
- warmup 后训练默认关闭 distill，使训练日志 loss 更接近纯 CE。

最终 VHA config 应包含：

```json
{
  "attn_type": "vha",
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "vha_enable_premix": false,
  "vha_enable_postmix": true,
  "vha_postmix_rank": 4
}
```

## 固定输入

| 名称 | 路径 |
| --- | --- |
| GQA checkpoint | `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged` |
| GQA config | `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA` |
| 训练数据目录 | `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/datasets/fineweb-edu` |
| mmap basename | `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/datasets/fineweb-edu/qwen` |
| tokenizer | `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/config/qwen3/tokenizer` |

## 统一输出目录

默认实验名：`qwen3_vha_1p7B_clean_kv_postmix`

```text
/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_clean_kv_postmix
```

主要产物结构：

```text
output/qwen3_vha_1p7B_clean_kv_postmix/
  activation_cache_128x4096_merged.npz
  init/
    model-00001-of-00001.safetensors
    model.safetensors.index.json
    config.json
    conversion_diagnostics.json
    conversion_recipe.json
    conversion_parts/
  refine/
    model-00001-of-00001.safetensors
    model.safetensors.index.json
    config.json
  warmup/
  warmup.json
  init_logits_loss.json
  refine_logits_loss.json
  logs/
```

## Stage 输入输出

### clean

执行：

```bash
bash scripts/run_vha_pipeline.sh clean
```

功能：删除当前新流程工作目录。

输入：无。

输出：删除 `output/qwen3_vha_1p7B_clean_kv_postmix`，不删除旧实验和 GQA 源模型。

### collect

执行：

```bash
bash scripts/run_vha_pipeline.sh collect
```

功能：八卡并行运行 GQA forward，采集 Q/K/V/pre_o activation，合并成单个 merged cache，并删除临时 shard。

输入：GQA checkpoint、GQA config、fineweb-edu 数据。

输出：

```text
activation_cache_128x4096_merged.npz
logs/collect_rank*.log
```

默认参数：`NUM_GPUS=8`、`ACTIVATION_SAMPLES=128`、`SEQ_LENGTH=4096`、`ACTIVATION_BATCH_SIZE=1`。

覆盖示例：

```bash
ACTIVATION_SAMPLES=256 bash scripts/run_vha_pipeline.sh collect
```

### convert

执行：

```bash
bash scripts/run_vha_pipeline.sh convert
```

功能：从 merged cache 读取每层 activation，八卡按 layer 并行做 `kv_postmix_only` 转换，然后组装最终 init checkpoint。

输入：`activation_cache_128x4096_merged.npz`。

输出：

```text
init/model-00001-of-00001.safetensors
init/model.safetensors.index.json
init/config.json
init/conversion_diagnostics.json
init/conversion_recipe.json
init/conversion_parts/layer0.pkl ... layer27.pkl
logs/convert_rank*.log
logs/convert_assemble.log
```

默认参数：`CONVERSION_MAX_TOKENS=65536`。

覆盖示例：

```bash
CONVERSION_MAX_TOKENS=131072 bash scripts/run_vha_pipeline.sh convert
```

### eval-init

执行：

```bash
bash scripts/run_vha_pipeline.sh eval-init
```

功能：在同一批真实 token 上比较 GQA teacher 和 init VHA student，输出 CE、CE gap 和 teacher→student logits KL。

输入：`init/` checkpoint。

输出：

```text
init_logits_loss.json
logs/init_eval.log
```

默认参数：`EVAL_SAMPLES=1`、`EVAL_SEQ_LENGTH=256`。

覆盖示例：

```bash
EVAL_SAMPLES=4 EVAL_SEQ_LENGTH=1024 bash scripts/run_vha_pipeline.sh eval-init
```

核心指标：

- `teacher_ce`
- `student_ce`
- `ce_gap = student_ce - teacher_ce`
- `teacher_to_student_kl`

### refine

执行：

```bash
bash scripts/run_vha_pipeline.sh refine
```

功能：从 init checkpoint 出发，使用 GQA teacher 做 cascading refine。默认训练 attention + norm 参数，减少初始转换误差。

输入：`init/` checkpoint、GQA teacher、fineweb mmap。

输出：

```text
refine/model-00001-of-00001.safetensors
refine/model.safetensors.index.json
refine/config.json
logs/refine.log
```

默认参数：`REFINE_SAMPLES=256`、`REFINE_SEQ_LENGTH=1024`、`REFINE_STEPS=200`、`REFINE_BATCH_SIZE=2`、`train_window=2`、`train_mode=attn_norm`。

覆盖示例：

```bash
REFINE_STEPS=1000 REFINE_SAMPLES=1024 bash scripts/run_vha_pipeline.sh refine
```

### eval-refine

执行：

```bash
bash scripts/run_vha_pipeline.sh eval-refine
```

功能：用和 `eval-init` 相同口径评估 refine 后 checkpoint。

输入：`refine/` checkpoint。

输出：

```text
refine_logits_loss.json
logs/refine_eval.log
```

判断方式：比较 `refine_logits_loss.json` 和 `init_logits_loss.json`。

### make-warmup-config

执行：

```bash
bash scripts/run_vha_pipeline.sh make-warmup-config
```

功能：生成干净 warmup 后训练配置。

输入：`refine/` checkpoint。

输出：`warmup.json`。

默认设置：`distill_alpha=0.0`、`layer_distill_beta=0.0`，使训练日志接近纯 CE。

如需开启 logit distill：

```bash
WARMUP_DISTILL_ALPHA=0.5 bash scripts/run_vha_pipeline.sh make-warmup-config
```

如需开启 layer distill：

```bash
WARMUP_LAYER_BETA=0.1 bash scripts/run_vha_pipeline.sh make-warmup-config
```

### warmup

执行：

```bash
bash scripts/run_vha_pipeline.sh warmup
```

功能：调用 `VHA/scripts/train_warmup.sh` 启动完整后训练。

输入：`warmup.json`、`refine/` checkpoint。

输出：`warmup/` 训练目录、checkpoint、trainer 日志。

注意：这是长训入口，建议先确认 init/refine CE/KL 合理后再启动。

### all

执行：

```bash
bash scripts/run_vha_pipeline.sh all
```

功能：顺序执行 `collect -> convert -> eval-init -> refine -> eval-refine -> make-warmup-config`。

输出：完整生成 activation cache、init checkpoint、init CE/KL、refine checkpoint、refine CE/KL、warmup 配置。

注意：`all` 不会自动启动 `warmup` 长训。

## 脚本输入输出

### `scripts/convert_gqa_to_vha_activation.py`

执行：

采集：

```bash
python scripts/convert_gqa_to_vha_activation.py   --gqa_checkpoint <gqa_model_state_merged>   --gqa_model_config <gqa_config_dir>   --output_path <init_dir>   --calib_data <fineweb_edu_dir>   --activation_cache <shard.npz>   --save_activation_cache --refresh_activation_cache   --collect_only --conversion_mode kv_postmix_only
```

转换：

```bash
python scripts/convert_gqa_to_vha_activation.py   --gqa_checkpoint <gqa_model_state_merged>   --gqa_model_config <gqa_config_dir>   --output_path <init_dir>   --activation_cache <merged.npz>   --conversion_mode kv_postmix_only   --conversion_layer_rank 0 --conversion_layer_count 8   --conversion_part_dir <part_dir>
```

组装：

```bash
python scripts/convert_gqa_to_vha_activation.py   --gqa_checkpoint <gqa_model_state_merged>   --gqa_model_config <gqa_config_dir>   --output_path <init_dir>   --activation_cache <merged.npz>   --conversion_mode kv_postmix_only   --assemble_from_parts <part_dir>
```

功能：采集 GQA activation，执行 KV 8→2 压缩，保持 16 Q heads，禁用 premix，初始化 postmix，输出 VHA init checkpoint。

输入：

- `--gqa_checkpoint`：GQA safetensors 目录。
- `--gqa_model_config`：GQA config 目录。
- `--calib_data`：fineweb-edu 数据目录。
- `--activation_cache`：collect 时为 shard 输出，convert 时为 merged cache 输入。
- `--conversion_mode kv_postmix_only`：当前唯一推荐模式。
- `--conversion_max_tokens`：每层转换 token 上限。
- `--conversion_layer_rank/count`：多进程按 layer 并行。

输出：

- activation shard：`activation_shard{rank}.npz`，包含 `layer0.Q` ... `layer27.pre_o`、`__meta__`。
- merged cache：由主控脚本合并成 `activation_cache_128x4096_merged.npz`。
- part：`conversion_parts/layer0.pkl ... layer27.pkl`，每个包含 `P_k`、`P_v`、`P_q`、`premix`、`postmix_U`、`postmix_V`、`diagnostics`。
- checkpoint：`model-00001-of-00001.safetensors`、`model.safetensors.index.json`、`config.json`、`conversion_diagnostics.json`、`conversion_recipe.json`。

### `scripts/eval_vha_logits_loss.py`

执行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vha_logits_loss.py   --gqa_checkpoint <gqa_model_state_merged>   --vha_checkpoint <vha_checkpoint_dir>   --gqa_model_config <gqa_config_dir>   --vha_model_config <vha_checkpoint_dir>   --data_path <qwen_mmap_basename>   --num_samples 1 --seq_length 256   --output_json <result.json>
```

功能：在同一批真实 tokens 上评估 GQA teacher 和 VHA student 的 logits 级差异。当前统一指标以该脚本输出的 CE/KL 为准。

输入：

- `--gqa_checkpoint`、`--gqa_model_config`：GQA teacher。
- `--vha_checkpoint`、`--vha_model_config`：VHA student。
- `--data_path`：mmap dataset basename，不带 `.bin/.idx`。
- `--num_samples`、`--seq_length`、`--batch_size`。
- `--kl_temperature`：默认 `1.0`。

输出：

```json
{
  "num_samples": 1,
  "seq_length": 256,
  "teacher_ce": 2.742619,
  "student_ce": 8.115311,
  "ce_gap": 5.372692,
  "teacher_to_student_kl": 5.569886,
  "batches": []
}
```

字段含义：

- `teacher_ce`：GQA teacher 在真实 label 上的 CE。
- `student_ce`：VHA student 在同一 label 上的 CE。
- `ce_gap`：`student_ce - teacher_ce`。
- `teacher_to_student_kl`：`KL(teacher || student)`。

### `scripts/refine_vha_cascading.py`

执行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/refine_vha_cascading.py   --single_process   --gqa_checkpoint <gqa_model_state_merged>   --gqa_model_config <gqa_config_dir>   --vha_checkpoint <init_checkpoint_dir>   --vha_model_config <init_checkpoint_dir>   --output_path <refine_output_dir>   --data_path <qwen_mmap_basename>   --num_samples 256 --seq_length 1024   --refine_steps 200 --batch_size 2   --train_window 2 --segment_stride 1 --train_mode attn_norm
```

功能：从 init checkpoint 出发，以 GQA teacher 为目标做 cascading refine，默认训练 attention + norm 参数，输出更适合 warmup 的 refined checkpoint。

输入：

- `--gqa_checkpoint`、`--gqa_model_config`。
- `--vha_checkpoint`、`--vha_model_config`。
- `--data_path`。
- refine 超参：`num_samples`、`seq_length`、`refine_steps`、`batch_size`、`train_window`、`train_mode`。单卡推荐加 `--single_process`。

输出：

```text
refine/model-00001-of-00001.safetensors
refine/model.safetensors.index.json
refine/config.json
```

日志由主控脚本写到 `logs/refine.log`。

### `VHA/scripts/train_warmup.sh`

执行：

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA
mpirun bash scripts/train_warmup.sh   /root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_clean_kv_postmix/warmup.json
```

功能：启动完整后训练。

输入：

- `warmup.json`。当前 clean pipeline 默认让 `distill_alpha=0.0`、`layer_distill_beta=0.0`，避免训练日志混合 loss 与纯 CE 混淆。

输出：

```text
warmup/checkpoint-*/
warmup/model_state/
warmup/trainer-*/
warmup/trainer_state.json
warmup/train_results.json
```

## 推荐执行顺序

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup
bash scripts/run_vha_pipeline.sh clean
bash scripts/run_vha_pipeline.sh all
# 查看 init_logits_loss.json 和 refine_logits_loss.json
bash scripts/run_vha_pipeline.sh warmup
```

## 关键环境变量

| 变量 | 默认值 | 功能 |
| --- | --- | --- |
| `EXP_NAME` | `qwen3_vha_1p7B_clean_kv_postmix` | 实验名与输出目录名 |
| `NUM_GPUS` | `8` | collect 和 convert 使用的 GPU 数 |
| `ACTIVATION_SAMPLES` | `128` | activation 采集样本数 |
| `CONVERSION_MAX_TOKENS` | `65536` | 每层转换使用的最大 token 数 |
| `EVAL_SAMPLES` | `1` | CE/KL 评估样本数 |
| `EVAL_SEQ_LENGTH` | `256` | CE/KL 评估长度 |
| `REFINE_STEPS` | `200` | cascading refine 每段步数 |
| `REFINE_SAMPLES` | `256` | refine 使用样本数 |
| `WARMUP_MAX_STEPS` | `12000` | warmup 后训练步数 |
| `WARMUP_DISTILL_ALPHA` | `0.0` | logits KL 蒸馏权重 |
| `WARMUP_LAYER_BETA` | `0.0` | layer distill 权重 |

## 成功标准

- `init/config.json` 中 `num_key_value_heads=2`、`vha_enable_premix=false`、`vha_enable_postmix=true`。
- `init/conversion_parts/` 下有 28 个 `layer*.pkl`。
- `init_logits_loss.json` 和 `refine_logits_loss.json` 都存在。
- refine 后 `student_ce`、`ce_gap` 或 `teacher_to_student_kl` 应优于 init。
- `warmup.json` 指向 `refine/` checkpoint。

## 不再作为主入口的脚本

以下脚本保留在仓库中，但不作为 clean pipeline 主入口：

| 脚本 | 状态 | 原因 |
| --- | --- | --- |
| `scripts/convert_gqa_to_vha.py` | 旧转换脚本 | 不使用 activation，不能满足当前目标 |
| `scripts/eval_vha_attention_error.py` | 旧 attention probe | 只看 attention 局部误差，旧版本有加载 bug，不作为主指标 |
| `scripts/eval_vha_init_loss.py` | 旧评估脚本 | 与 attention probe 类似，不作为当前统一指标 |
| `scripts/refine_vha_per_layer.py` | 旧 per-layer refine | 当前使用 cascading refine |

当前统一判断指标以 `eval_vha_logits_loss.py` 输出的 CE/KL 为准。
