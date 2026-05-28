"""
Distillation losses for VHA warmup training.

1. DistillationLoss: Logit-level KL divergence (wraps LanguageLoss)
2. LayerAttnDistillation: Per-layer attention output MSE alignment via hooks

The layer-wise loss is computed in training_step (outside forward_backward_pipeline)
to avoid issues with recompute. Teacher and student hidden states are captured
during their respective forward passes.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss


# =============================================================================
# Layer-wise Attention Distillation
# =============================================================================

class LayerAttnDistillation:
    """
    Per-layer hidden state alignment between teacher (GQA) and student (VHA).

    Hooks on TransformerLayer (full layer output = attention + MLP + residual),
    not just self_attn. This allows MLP to compensate for attention pattern differences.

    Integration: The layer loss is added to the total loss in DistillationLoss.forward_impl
    via the class-level _layer_loss buffer.
    """

    # Class-level buffer for the computed layer loss (picked up by DistillationLoss)
    _layer_loss: Tensor | None = None
    _instance: 'LayerAttnDistillation | None' = None

    def __init__(self, student_model, teacher_model, beta: float = 0.1):
        self.beta = beta
        self._teacher_outputs = {}
        self._student_outputs = {}
        self._hooks = []
        self.num_layers = 0
        self._capturing = False  # guard against recompute double-capture
        self._register_hooks(student_model, teacher_model)
        LayerAttnDistillation._instance = self

    def _register_hooks(self, student_model, teacher_model):
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        student_layers = [l for l in student_model.layers if isinstance(l, TransformerLayer)]
        teacher_layers = [l for l in teacher_model.layers if isinstance(l, TransformerLayer)]

        assert len(student_layers) == len(teacher_layers), (
            f"Layer mismatch: student={len(student_layers)}, teacher={len(teacher_layers)}"
        )
        self.num_layers = len(student_layers)

        for idx, (s_layer, t_layer) in enumerate(zip(student_layers, teacher_layers)):
            # Hook on full TransformerLayer output (after attention + MLP + residual)
            h = t_layer.register_forward_post_hook(
                self._make_hook(self._teacher_outputs, idx, detach=True)
            )
            self._hooks.append(h)
            h = s_layer.register_forward_post_hook(
                self._make_hook(self._student_outputs, idx, detach=False)
            )
            self._hooks.append(h)

    def _make_hook(self, buffer_dict, layer_idx, detach=True):
        """Create hook that respects the _capturing flag."""
        parent = self

        def hook_fn(module, input, output):
            if not parent._capturing:
                return
            # TransformerLayer returns dict or tensor
            if isinstance(output, dict):
                out = output.get("hidden_states", output.get("output", None))
                if out is None:
                    # Try first value
                    out = next(iter(output.values()))
            elif isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            buffer_dict[layer_idx] = out.detach() if detach else out

        return hook_fn

    def begin_capture(self):
        """Enable hook capture. Call before teacher+student forward."""
        self._capturing = True
        self._teacher_outputs.clear()
        self._student_outputs.clear()

    def end_capture_and_compute(self) -> Tensor:
        """Disable capture and compute layer-wise MSE loss."""
        self._capturing = False

        if not self._student_outputs or not self._teacher_outputs:
            LayerAttnDistillation._layer_loss = None
            return paddle.zeros([1])

        total_mse = paddle.zeros([1])
        n = 0
        for idx in range(self.num_layers):
            if idx in self._student_outputs and idx in self._teacher_outputs:
                s_out = self._student_outputs[idx]
                t_out = self._teacher_outputs[idx]
                min_len = min(s_out.shape[1], t_out.shape[1])
                mse = F.mse_loss(
                    s_out[:, :min_len, :].cast("float32"),
                    t_out[:, :min_len, :].cast("float32"),
                )
                total_mse += mse
                n += 1

        self._student_outputs.clear()
        self._teacher_outputs.clear()

        if n == 0:
            LayerAttnDistillation._layer_loss = None
            return paddle.zeros([1])

        loss = self.beta * (total_mse / n)
        LayerAttnDistillation._layer_loss = loss
        return loss

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =============================================================================
# Logit Distillation Loss
# =============================================================================

class DistillationLoss(LanguageLoss):
    """
    Language loss + KL divergence distillation + optional layer-wise MSE.

    Usage:
        1. Before student forward: DistillationLoss.set_teacher_logits(logits)
        2. Student forward computes loss normally + KL term + layer term
        3. After backward: buffers are auto-cleared
    """

    _teacher_logits: Tensor | None = None
    _alpha: float = 0.5
    _temperature: float = 2.0
    _scale: float = 1.0  # Distillation scale factor (1.0=full, 0.0=off)

    @classmethod
    def set_teacher_logits(cls, logits: Tensor | None):
        cls._teacher_logits = logits

    @classmethod
    def set_scale(cls, scale: float):
        """Set distillation scale (for linear decay). 0 disables distillation."""
        cls._scale = scale

    @classmethod
    def configure(cls, alpha: float = 0.5, temperature: float = 2.0):
        cls._alpha = alpha
        cls._temperature = temperature

    def forward_impl(self, logits: Tensor | tuple, labels: Tensor) -> Tensor:
        """Compute CE loss + KL distillation + layer-wise MSE."""
        ce_loss = super().forward_impl(logits, labels)

        if self._teacher_logits is None:
            return ce_loss

        if isinstance(logits, tuple):
            return ce_loss

        T = self._temperature
        alpha = self._alpha * self._scale

        teacher_logits = self._teacher_logits
        s_len = logits.shape[1]
        t_len = teacher_logits.shape[1]
        min_len = min(s_len, t_len)

        labels_trimmed = labels[:, :min_len]
        valid_mask = (labels_trimmed != self.ignored_index).cast("float32")
        n_valid = valid_mask.sum()

        if n_valid == 0:
            DistillationLoss._teacher_logits = None
            return ce_loss

        # Subbatch KL computation
        chunk_size = self.loss_subbatch_sequence_length if self.use_subbatch else min_len
        kl_sum = paddle.zeros([1])
        for start in range(0, min_len, chunk_size):
            end = min(start + chunk_size, min_len)
            s_chunk = logits[:, start:end, :].cast("float32") / T
            t_chunk = teacher_logits[:, start:end, :].cast("float32") / T
            mask_chunk = valid_mask[:, start:end]

            s_log_probs = F.log_softmax(s_chunk, axis=-1)
            t_probs = F.softmax(t_chunk, axis=-1)

            kl_chunk = F.kl_div(s_log_probs, t_probs, reduction='none')
            kl_chunk = kl_chunk.sum(axis=-1)
            kl_sum += (kl_chunk * mask_chunk).sum()

        kl = kl_sum / n_valid * (T * T)

        # Combined: CE + KL + layer-wise MSE
        loss = (1 - alpha) * ce_loss + alpha * kl

        # Add layer-wise attention MSE if available (scaled by distill_scale)
        # By this point all student hooks have fired, so we can compute the layer loss
        if LayerAttnDistillation._instance is not None:
            layer_loss = LayerAttnDistillation._instance.end_capture_and_compute()
            loss = loss + layer_loss * self._scale

        DistillationLoss._teacher_logits = None
        return loss
