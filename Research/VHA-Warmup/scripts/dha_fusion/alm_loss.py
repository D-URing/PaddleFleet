"""ALM (Augmented Lagrangian Method) loss for DHA-style head fusion.

Constraint: within each group of source KV heads, post-K-proj activations
should converge to a common vector. Trainer minimizes:

    L_total = L_lm + lambda * max(L_fusion - t(s), 0)

with a decaying target t(s) and dual update on lambda. As t(s) -> 0 and
lambda grows, intra-group activations are forced to coincide; once converged,
omega becomes irrelevant and the heads can be folded into one with no loss.

L_fusion = (1 / Z) * sum over layers, groups, heads in group of
              ||K_h - K_group_mean||^2 / ||K_group_mean||^2  +  same for V
where the divisor normalizes scale (relative MSE).

Notes:
- Constraint is computed on per-token, per-batch activations from the most
  recent forward pass (cached on each FusionState).
- The trainer must call `compute_constraint(...)` after the forward, then
  combine with LM loss before backward.
- Dual update happens once per training step using the *detached* constraint value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import paddle


@dataclass
class ALMConfig:
    # ---- target schedule t(s) ----
    target_initial: float = 1.0   # t_0; will be auto-tuned to first observed L_fusion if <0
    target_decay_steps: int = 1500  # linear decay to 0 over this many steps
    # After target_decay_steps, t(s) stays at 0 -> hard constraint
    auto_tune_initial: bool = True  # set t_0 = first measured L_fusion if True

    # ---- dual update on lambda ----
    lambda_init: float = 0.0
    lambda_lr: float = 1.0       # eta in lambda <- lambda + eta * max(C - t, 0)
    lambda_max: float = 100.0    # clip
    # Update lambda only every K steps (smooths noisy gradients)
    dual_update_interval: int = 50

    # ---- constraint smoothing ----
    # Whether to use squared (quadratic penalty) plus linear (Lagrangian) terms.
    use_quadratic_penalty: bool = False
    mu: float = 0.0  # quadratic coefficient if use_quadratic_penalty


class ALMState:
    """Mutable ALM state: lambda, current target, step counter, history."""

    def __init__(self, cfg: ALMConfig):
        self.cfg = cfg
        self.lam: float = cfg.lambda_init
        self.target_initial: float = cfg.target_initial
        self.step: int = 0
        self._first_constraint_seen: float | None = None
        # Step-wise log; trainer can dump to JSON
        self.history: List[dict] = []

    # ----------------------------------------------------------------- t(s)
    def current_target(self) -> float:
        # If auto-tune hasn't run yet (target_initial <= 0), behave as t=0 so
        # initial violation = C and lam=0 gives total = lm_loss.
        t0 = self.target_initial
        if t0 <= 0:
            return 0.0
        s = self.step
        S = self.cfg.target_decay_steps
        if s >= S:
            return 0.0
        return t0 * max(0.0, 1.0 - s / max(1, S))

    # -------------------------------------------------------- dual update
    def maybe_update_lambda(self, constraint_value: float) -> bool:
        """Apply lambda <- clip(lambda + eta * max(C - t, 0), 0, lambda_max).

        Returns True if updated this step.
        """
        cfg = self.cfg
        # Skip step 0: target_initial not yet auto-tuned (would over-update).
        if self.step == 0:
            return False
        if self.step % cfg.dual_update_interval != 0:
            return False
        t = self.current_target()
        violation = max(0.0, constraint_value - t)
        new_lam = self.lam + cfg.lambda_lr * violation
        new_lam = max(0.0, min(cfg.lambda_max, new_lam))
        self.lam = new_lam
        return True

    # --------------------------------------------------------- bookkeeping
    def end_step(self, constraint_value: float, lm_loss_value: float, log_extra: dict | None = None) -> None:
        if self.cfg.auto_tune_initial and self._first_constraint_seen is None:
            self._first_constraint_seen = constraint_value
            # Set t_0 = first observed constraint so initial violation is zero
            # (gentle ramp). If user provided target_initial > 0, override only if auto.
            self.target_initial = max(constraint_value, 1e-8)
        record = {
            "step": self.step,
            "constraint": constraint_value,
            "target": self.current_target(),
            "lambda": self.lam,
            "lm_loss": lm_loss_value,
            "violation": max(0.0, constraint_value - self.current_target()),
        }
        if log_extra:
            record.update(log_extra)
        self.history.append(record)
        self.step += 1


# ============================================================================
# Constraint computation
# ============================================================================
def compute_layer_constraint(
    k_pre: paddle.Tensor,
    v_pre: paddle.Tensor,
    fusion_mask: paddle.Tensor,
    group_size: paddle.Tensor,
) -> paddle.Tensor:
    """Vectorized relative intra-group MSE for a single layer (no Python loop).

    Args:
        k_pre, v_pre: [B, T, n_src, d] activations (cached pre-fusion).
        fusion_mask: [n_src, n_groups] one-hot (head -> group).
        group_size:  [n_groups] float, count of heads per group.

    Returns:
        Scalar tensor: mean over (K,V) and groups of
            E_btd[(k_h - k_group_mean)^2] / E_btd[(k_group_mean)^2]
        Numerically equivalent to the original Python-loop form for equal-size groups.
    """
    eps = 1e-8
    mask = fusion_mask.cast(k_pre.dtype)                     # [H, G]
    gs = group_size.cast(k_pre.dtype).reshape([1, 1, -1, 1]) # [1,1,G,1]

    # Group means: [B, T, G, D]
    k_gsum = paddle.einsum("bthd,hg->btgd", k_pre, mask)
    v_gsum = paddle.einsum("bthd,hg->btgd", v_pre, mask)
    k_gmean = k_gsum / gs
    v_gmean = v_gsum / gs

    # Per-head deviation: dev[b,t,h,d] = k[b,t,h,d] - k_gmean[b,t,group(h),d]
    k_dev = k_pre - paddle.einsum("hg,btgd->bthd", mask, k_gmean)
    v_dev = v_pre - paddle.einsum("hg,btgd->bthd", mask, v_gmean)

    # Per-group dev^2 averaged over (B,T,D) and over heads-in-group:
    # dev2_per_group[b,t,g,d] = sum_h mask[h,g] * dev[b,t,h,d]^2 / |g|
    k_dev2_g = paddle.einsum("bthd,hg->btgd", k_dev * k_dev, mask) / gs
    v_dev2_g = paddle.einsum("bthd,hg->btgd", v_dev * v_dev, mask) / gs
    k_dev2 = k_dev2_g.mean(axis=[0, 1, 3])  # [G]
    v_dev2 = v_dev2_g.mean(axis=[0, 1, 3])  # [G]

    k_scale = (k_gmean * k_gmean).mean(axis=[0, 1, 3]) + eps  # [G]
    v_scale = (v_gmean * v_gmean).mean(axis=[0, 1, 3]) + eps

    return ((k_dev2 / k_scale).mean() + (v_dev2 / v_scale).mean()) * 0.5


def compute_total_constraint(fusion_states) -> paddle.Tensor:
    """Sum over all layers; expects each FusionState has cached_k_pre_fusion,
    cached_v_pre_fusion populated by the most recent forward pass.

    Returns scalar tensor; gradient flows back through cached K, V into K_proj/V_proj.
    """
    per_layer: List[paddle.Tensor] = []
    for fs in fusion_states:
        if fs.cached_k_pre_fusion is None or fs.cached_v_pre_fusion is None:
            raise RuntimeError(
                "cached_k/v_pre_fusion is None - did the forward pre_hook fire?"
            )
        # Native dtype constraint compute (bf16 ok for relative MSE; final reduction is fp32)
        c = compute_layer_constraint(
            fs.cached_k_pre_fusion,
            fs.cached_v_pre_fusion,
            fs.fusion_mask,
            fs.group_size,
        ).cast("float32")
        per_layer.append(c)
    return paddle.stack(per_layer).mean()


# ============================================================================
# ALM total loss assembly
# ============================================================================
def alm_combine(
    lm_loss: paddle.Tensor,
    constraint: paddle.Tensor,
    state: ALMState,
) -> paddle.Tensor:
    """Build the total ALM-augmented loss tensor.

        L_total = L_lm + lambda * max(C - t, 0)             (Lagrangian)
                + (mu/2) * max(C - t, 0)^2  if quadratic    (Augmented)

    Note: lambda is treated as a scalar coefficient (not a tensor with grad).
    Dual update is handled separately via state.maybe_update_lambda(C.item()).
    """
    cfg = state.cfg
    t = state.current_target()
    violation = paddle.clip(constraint - t, min=0.0)
    total = lm_loss + state.lam * violation
    if cfg.use_quadratic_penalty and cfg.mu > 0:
        total = total + (cfg.mu / 2.0) * (violation ** 2)
    return total
