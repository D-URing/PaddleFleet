"""
Distillation losses for VHA warmup training.

1. DistillationLoss: CE + additive logit-level KL divergence.
2. LayerAttnDistillation: Per-layer attention output MSE alignment via hooks.
"""

import paddle
import paddle.nn.functional as F
from paddle import Tensor

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss


class LayerAttnDistillation:
    """
    Per-layer attention-output alignment between teacher (GQA) and student (VHA).

    Hooks on each TransformerLayer.self_attn output before residual/MLP, so the loss
    directly supervises the attention module affected by KV-head compression.
    """

    _layer_loss: Tensor | None = None
    _instance: "LayerAttnDistillation | None" = None

    def __init__(self, student_model, teacher_model, beta: float = 0.1):
        self.beta = beta
        self._teacher_outputs = {}
        self._student_outputs = {}
        self._hooks = []
        self.num_layers = 0
        self._capturing = False
        self._register_hooks(student_model, teacher_model)
        LayerAttnDistillation._instance = self

    def _register_hooks(self, student_model, teacher_model):
        def find_transformer_layers(model):
            layers = []
            for _, sublayer in model.named_sublayers():
                if hasattr(sublayer, "self_attn") and hasattr(sublayer.self_attn, "qkv_proj"):
                    layers.append(sublayer)
            return layers

        student_layers = find_transformer_layers(student_model)
        teacher_layers = find_transformer_layers(teacher_model)

        if len(student_layers) != len(teacher_layers):
            raise ValueError(f"Layer mismatch: student={len(student_layers)}, teacher={len(teacher_layers)}")
        if not student_layers:
            raise ValueError("No TransformerLayer.self_attn modules found for attention-output distillation")

        self.num_layers = len(student_layers)

        for idx, (s_layer, t_layer) in enumerate(zip(student_layers, teacher_layers)):
            h = t_layer.self_attn.register_forward_post_hook(
                self._make_hook(self._teacher_outputs, idx, detach=True)
            )
            self._hooks.append(h)
            h = s_layer.self_attn.register_forward_post_hook(
                self._make_hook(self._student_outputs, idx, detach=False)
            )
            self._hooks.append(h)

    def _make_hook(self, buffer_dict, layer_idx, detach=True):
        parent = self

        def hook_fn(module, input, output):
            if not parent._capturing:
                return
            if isinstance(output, dict):
                out = output.get("hidden_states", output.get("output", None))
                if out is None:
                    out = next(iter(output.values()))
            elif isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            buffer_dict[layer_idx] = out.detach() if detach else out

        return hook_fn

    def begin_capture(self):
        self._capturing = True
        self._teacher_outputs.clear()
        self._student_outputs.clear()

    def end_capture_and_compute(self) -> Tensor:
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


class DistillationLoss(LanguageLoss):
    """
    CE + additive KL divergence + optional attention-output MSE.

    CE coefficient is always 1.0. Distillation terms are auxiliary and can be
    linearly decayed by DistillationLoss.set_scale().
    """

    _teacher_logits: Tensor | None = None
    _alpha: float = 1.0
    _temperature: float = 1.0
    _scale: float = 1.0

    @classmethod
    def set_teacher_logits(cls, logits: Tensor | None):
        cls._teacher_logits = logits

    @classmethod
    def set_scale(cls, scale: float):
        cls._scale = scale

    @classmethod
    def configure(cls, alpha: float = 1.0, temperature: float = 1.0):
        cls._alpha = alpha
        cls._temperature = temperature

    def forward_impl(self, logits: Tensor | tuple, labels: Tensor) -> Tensor:
        ce_loss = super().forward_impl(logits, labels)

        if self._teacher_logits is None:
            return ce_loss

        if isinstance(logits, tuple):
            return ce_loss

        temperature = self._temperature
        kl_weight = self._alpha * self._scale

        teacher_logits = self._teacher_logits
        s_len = logits.shape[1]
        t_len = teacher_logits.shape[1]
        min_len = min(s_len, t_len)

        valid_mask = (labels[:, :min_len] != self.ignored_index).cast("float32")
        n_valid = valid_mask.sum()

        if n_valid == 0:
            DistillationLoss._teacher_logits = None
            return ce_loss

        chunk_size = self.loss_subbatch_sequence_length if self.use_subbatch else min_len
        kl_sum = paddle.zeros([1])
        for start in range(0, min_len, chunk_size):
            end = min(start + chunk_size, min_len)
            student_chunk = logits[:, start:end, :].cast("float32") / temperature
            teacher_chunk = teacher_logits[:, start:end, :].cast("float32") / temperature
            mask_chunk = valid_mask[:, start:end]

            student_log_probs = F.log_softmax(student_chunk, axis=-1)
            teacher_probs = F.softmax(teacher_chunk, axis=-1)

            kl_chunk = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(axis=-1)
            kl_sum += (kl_chunk * mask_chunk).sum()

        kl = kl_sum / n_valid * (temperature * temperature)
        loss = ce_loss + kl_weight * kl

        if LayerAttnDistillation._instance is not None:
            layer_loss = LayerAttnDistillation._instance.end_capture_and_compute()
            loss = loss + layer_loss * self._scale

        if paddle.distributed.get_rank() == 0:
            print(
                "[DISTILL_LOSS] "
                f"ce={float(ce_loss.numpy()):.6f} "
                f"kl={float(kl.numpy()):.6f} "
                f"kl_weight={float(kl_weight):.6f} "
                f"layer={float(layer_loss.numpy() if LayerAttnDistillation._instance is not None else 0.0):.6f} "
                f"total={float(loss.numpy()):.6f}",
                flush=True,
            )

        DistillationLoss._teacher_logits = None
        return loss
