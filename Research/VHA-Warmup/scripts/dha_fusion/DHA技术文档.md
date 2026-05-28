# DHA (Dynamic Head Attention) 头融合方案文档

## 一、DHA原论文核心思想

### 1.1 问题背景

DHA (Dynamic Head Attention) 解决的核心问题是：如何在保持模型性能的前提下，将多KV头的GQA模型高效转换为更少KV头的VHA模型。

具体场景：
- **起点**: GQA模型，num_key_value_heads=8，总参数量大
- **终点**: VHA模型，num_key_value_heads=2，通过postmix补偿精度损失
- **挑战**: 直接转换会导致精度显著下降

### 1.2 核心机制

DHA通过以下机制实现渐进式头融合：

#### 1.2.1 Omega门控融合

每层的每个KV头引入可学习的权重矩阵 ω_K 和 ω_V：

```
ω_K ∈ [n_src_heads, n_groups]  # 8 x 2
ω_V ∈ [n_src_heads, n_groups]  # 8 x 2
```

融合操作：
```
K_fused[b, t, g, d] = Σ_h ω_K[h, g] * K[b, t, h, d]  # g ∈ [0,1], h ∈ group(g)
V_fused[b, t, g, d] = Σ_h ω_V[h, g] * V[b, t, h, d]
```

#### 1.2.2 分组策略

基于余弦相似度的 exhaustive search 找到最优分组：

```
groupings[layer_idx] = [group_0_heads, group_1_heads]
例如：groupings[0] = [[0, 2, 6, 7], [1, 3, 4, 5]]
```

分组依据：
- 计算每层所有K头和V头之间的余弦相似度
- 搜索使得分组误差最小的分组方案
- 确保每组大小均衡（4头一组）

#### 1.2.3 Postmix补偿

由于8→2的KV收缩会损失rank，引入低秩残差补偿：

```
postmix_U ∈ [n_q_heads, rank]     # 16 x 4
postmix_V ∈ [n_q_heads, rank]     # 16 x 4

delta = attention_output @ U @ V  # 低秩残差
output = attention_output + delta
```

### 1.3 增强拉格朗日方法 (ALM)

DHA使用ALM将头融合约束转化为优化目标：

#### 目标函数

```
L_total = L_lm + λ(s) * max(C(s) - t(s), 0)
```

其中：
- `L_lm`: 原始语言模型损失
- `λ(s)`: 拉格朗日乘子（随步数递增）
- `C(s)`: 头融合约束值（内组KV相对MSE）
- `t(s)`: 目标值（随步数递减，从初始值衰减到0）

#### 约束计算

```
C = (1/Z) * Σ over layers, groups, heads in group of
    ||K_h - K_group_mean||^2 / ||K_group_mean||^2  +  same for V
```

使用相对MSE归一化，避免尺度影响。

#### 双变量更新

```
λ(s+1) = clip(λ(s) + η * max(C(s) - t(s), 0), 0, λ_max)
t(s+1) = t_0 * max(0, 1 - s / S)  # 线性衰减
```

## 二、方案设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GQA Checkpoint (起点)                      │
│                  num_key_value_heads = 8                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  DHA Fusion Training                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  每层附加参数:                                           │ │
│  │  - omega_k_logits: [8, 2]  # K头融合门控               │ │
│  │  - omega_v_logits: [8, 2]  # V头融合门控               │ │
│  │  - postmix_U: [16, 4]        # Postmix补偿U矩阵        │ │
│  │  - postmix_V: [16, 4]        # Postmix补偿V矩阵        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  Training: L_total = L_lm + λ * max(C - t, 0)               │
│  - 固定GQA模型权重                                          │ │
│  - 只训练omega和postmix参数                                  │ │
│  - 约2000步，LR=1e-5                                         │ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Fold: 融合收敛检查                            │
│  - 检查omega是否收敛为近one-hot                              │ │
│  - 如果收敛，折叠K_proj/V_proj权重 (8→2)                    │ │
│  - 保留postmix参数                                           │ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   VHA Checkpoint (终点)                       │
│                  num_key_value_heads = 2                       │
│                  + postmix_UV补偿结构                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Hook实现策略

DHA通过PyTorch的forward hook机制实现，不修改模型结构：

#### Pre-Hook (core_attention.forward_pre_hook)

```python
def pre_hook(layer, args):
    K, V = args[1], args[2]  # [B, T, 8, 128]

    # 缓存原始KV用于约束计算
    fs.cached_k_pre_fusion = K
    fs.cached_v_pre_fusion = V

    # 应用omega融合
    K_fused, V_fused = fs.fuse_kv(K, V)  # [B, T, 2, 128]

    # 修改args返回
    new_args = list(args)
    new_args[1] = K_fused
    new_args[2] = V_fused
    return tuple(new_args)
```

#### Post-Hook (core_attention.forward_post_hook)

```python
def post_hook(layer, args, output):
    attn_out = output[0]  # [B, T, 16*128]

    # 应用postmix补偿
    mixed = fs.apply_postmix(attn_out)

    return (mixed,) + output[1:]
```

### 2.3 分组初始化

#### 余弦相似度加权初始化

```python
logits[h, g] = T * mean_{m in members(g)} cos[h, m]
```

- T = 5.0 (temperature)
- 基于预转换时计算的K头/V头余弦相似度矩阵
- 使相似头更倾向于分到同一组

## 三、当前实现结构

### 3.1 目录结构

```
scripts/dha_fusion/
├── alm_loss.py              # ALM损失实现
├── attention_patch.py       # Hook实现和FusionState
├── grouping.py              # 分组加载和验证
├── compute_kv_cosine_grouping.py  # 余弦相似度计算
├── fold.py                  # 融合后折叠脚本
├── run_dha_fusion.py        # 训练入口
├── run_dha_fusion.sh        # 启动脚本
├── dha_fusion.json          # 完整配置
├── dha_fusion_smoke.json    # 快速测试配置
└── kv_cosine_groupings.json # 余弦相似度数据
```

### 3.2 核心类和函数

#### alm_loss.py

```python
@dataclass
class ALMConfig:
    target_initial: float = 1.0   # t_0，auto-tune到首次观测值
    target_decay_steps: int = 1500  # 目标衰减步数
    lambda_init: float = 0.0
    lambda_lr: float = 1.0
    lambda_max: float = 100.0
    dual_update_interval: int = 50

class ALMState:
    """管理λ和t的状态，记录历史"""
    def current_target(self) -> float
    def maybe_update_lambda(self, constraint_value: float) -> bool
    def end_step(self, constraint_value, lm_loss_value)

def compute_layer_constraint(k_pre, v_pre, fusion_mask, group_size)
def compute_total_constraint(fusion_states)
```

#### attention_patch.py

```python
class FusionState(nn.Layer):
    """每层的融合参数和缓存"""
    omega_k_logits: Parameter  # [n_src, n_groups]
    omega_v_logits: Parameter  # [n_src, n_groups]
    postmix_U: Parameter       # [n_q_heads, rank]
    postmix_V: Parameter       # [n_q_heads, rank]
    cached_k_pre_fusion: Tensor
    cached_v_pre_fusion: Tensor

    def fuse_kv(self, k, v) -> Tuple[Tensor, Tensor]
    def apply_postmix(self, attn_out) -> Tensor

def install_fusion_hooks(...)
def install_all_fusion_hooks(...)
```

#### run_dha_fusion.py

```python
class DHALanguageLoss(LanguageLoss):
    """LM loss + ALM constraint的in-graph实现"""
    def forward(self, logits, labels):
        lm_loss = super().forward(logits, labels)
        constraint = compute_total_constraint(self.fusion_states_holder)
        t = self.alm.current_target()
        violation = clip(constraint - t, min=0.0)
        total = lm_loss + self.alm.lam * violation
        return total

class ALMCallback(TrainerCallback):
    """更新λ和记录历史"""
    def on_step_end(self, args, state, control, **kwargs):
        # 定期更新λ
        # 记录C, t, λ, lm_loss等

def build_cosine_omega_init(cosine_path, groupings, temperature)
```

### 3.3 配置参数说明

#### 核心训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gqa_checkpoint` | - | GQA检查点路径 |
| `groupings_path` | - | 分组诊断文件路径 |
| `max_steps` | 2000 | 最大训练步数 |
| `lr` | 1e-5 | 学习率（常量） |
| `batch_size` | 4 | 批大小 |
| `seq_length` | 4096 | 序列长度 |

#### ALM约束参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fusion_decay_steps` | 1500 | 目标t衰减步数 |
| `lambda_init` | 0.0 | λ初始值 |
| `lambda_lr` | 1.0 | λ更新步长 |
| `lambda_max` | 100.0 | λ上限 |
| `dual_update_interval` | 50 | λ更新间隔 |

#### Omega初始化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `omega_init_path` | - | 余弦相似度文件路径 |
| `omega_init_temperature` | 5.0 | 余弦加权温度 |

### 3.4 模型配置

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "num_attention_heads": 16,
  "num_key_value_heads": 8,      // GQA起点
  "head_dim": 128,
  "num_hidden_layers": 28,
  "hidden_size": 2048,
  "intermediate_size": 6144,
  ...
}
```

### 3.5 训练流程

```
1. 构建GQA pipeline模型
   ├── 加载provider (qwen_provider)
   └── 创建DHALanguageLoss (含fusion_states_holder引用)

2. 加载GQA检查点权重

3. 安装fusion hooks
   ├── 加载groupings (每层2组，每组4头)
   ├── 可选: 加载cosine相似度数据用于omega初始化
   ├── 为每层创建FusionState
   │   ├── omega_k_logits: [8, 2]
   │   ├── omega_v_logits: [8, 2]
   │   ├── postmix_U: [16, 4]
   │   └── postmix_V: [16, 4]
   └── 注册forward_pre_hook和forward_post_hook

4. 训练
   ├── forward:
   │   ├── pre_hook缓存K/V，应用omega融合
   │   ├── 计算attention（使用融合后的2个KV头）
   │   ├── post_hook应用postmix补偿
   │   └── loss层从cached K/V计算约束C
   ├── backward: 梯度流回omega和postmix参数
   └── ALMCallback:
       ├── 每dual_interval步更新λ
       └── 每log_interval步记录C, t, λ

5. 保存checkpoint
   ├── 基础GQA权重
   ├── dha_fusion参数
   └── alm_history.json
```

## 四、当前问题分析

### 4.1 转换终点相悖问题

当前DHA方案存在两个目标，存在潜在冲突：

| 目标 | 描述 | 优化方向 |
|------|------|----------|
| **头融合收敛** | omega权重收敛为one-hot，C→0 | 增大λ，强化约束 |
| **Loss下降** | LM损失下降，保持模型性能 | 减小λ，保持LM优化为主 |

#### 相悖原因

1. **约束与目标函数的权衡**:
   - 增大λ可以加速头融合，但会干扰LM损失下降
   - 减小λ有利于LM损失，但头融合可能收敛缓慢

2. **Postmix补偿机制**:
   - Postmix补偿了rank损失，使得即使KV不完美融合也能维持较低LM loss
   - 这可能导致omega不需要严格收敛就能获得可接受的LM loss

### 4.2 当前参数配置

```json
{
  "fusion_decay_steps": 1500,      // t从初始值线性衰减到0
  "lambda_init": 0.0,
  "lambda_lr": 1.0,                // λ增长步长
  "lambda_max": 100.0,
  "dual_update_interval": 50,      // 每50步更新一次λ
  "max_steps": 2000
}
```

### 4.3 调整建议

#### 方案1: 两阶段训练

```
阶段1: 融合优先 (0-1000步)
  - 较大的λ_lr (2.0)
  - 较快的t衰减 (fusion_decay_steps=800)

阶段2: LM优化优先 (1000-2000步)
  - 较小的λ_lr (0.5)
  - t已经为0，只维持轻微约束
```

#### 方案2: 自适应λ更新

```python
# 基于C的收敛速度动态调整λ_lr
if C下降快:
    λ_lr *= 0.8  # 减缓λ增长
else:
    λ_lr *= 1.2  # 加速λ增长
```

#### 方案3: Postmix强度控制

```python
# 在融合初期减少postmix影响，迫使omega收敛
postmix_scale = min(1.0, step / 1000)
delta = postmix_scale * attention_output @ U @ V
```

## 五、参考资料

### 5.1 相关论文

- **DHA原论文**: Dynamic Head Attention for Efficient Transformer Models
- **TransMLA**: 基于PCA的低秩注意力转换（来源）

### 5.2 代码路径

- 实现代码: `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/scripts/dha_fusion/`
- GQA检查点: `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/`
- 分组诊断: `/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_kv_postmix_activation_128/conversion_diagnostics.json`

### 5.3 启动命令

```bash
# Smoke测试
bash run_dha_fusion.sh dha_fusion_smoke.json

# 完整训练
bash run_dha_fusion.sh dha_fusion.json
```

---

*文档生成时间: 2026-05-28*
*基于对话历史和代码实现整理*