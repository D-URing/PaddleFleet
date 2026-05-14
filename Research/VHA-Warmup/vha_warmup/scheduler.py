"""
Warm-up training scheduler for VHA.

Three-phase progressive unfreezing:
- Phase 1: Freeze Q_new, W2[0]=I. Train W2[1], KV, postmix, O_proj.
- Phase 2: Freeze W2[0]. Unfreeze Q_new.
- Phase 3: Unfreeze all.
"""

from typing import List, Optional


class WarmupScheduler:
    """Manages parameter freezing/unfreezing across warm-up phases."""

    def __init__(
        self,
        phase1_steps: int = 1000,
        phase2_steps: int = 2000,
        total_steps: Optional[int] = None,
    ):
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.total_steps = total_steps
        self._current_phase = 0

    def get_phase(self, step: int) -> int:
        """Return current phase (1, 2, or 3) based on step."""
        if step < self.phase1_steps:
            return 1
        elif step < self.phase1_steps + self.phase2_steps:
            return 2
        else:
            return 3

    def get_frozen_params(self, step: int) -> List[str]:
        """
        Return list of parameter name patterns to freeze at given step.

        Returns:
            List of glob patterns for parameters that should be frozen.
        """
        phase = self.get_phase(step)

        if phase == 1:
            return [
                "q_proj",       # Freeze physical Q
                "w2_rot",       # Freeze all W2 (W2[0]=I stays, W2[1] trained via w2_delta)
                # Actually: freeze q_proj, keep w2[0] as identity
                # Trainable: w2[1], k_proj, v_proj, o_proj, v4_U, v4_V
            ]
        elif phase == 2:
            return [
                "w2_rot.0",     # Keep W2[0] = I frozen
                # Trainable: q_proj, w2[1], k_proj, v_proj, o_proj, v4_U, v4_V
            ]
        else:
            return []  # All trainable

    def should_transition(self, step: int) -> bool:
        """Check if we just entered a new phase."""
        phase = self.get_phase(step)
        if phase != self._current_phase:
            self._current_phase = phase
            return True
        return False

    def apply_freeze(self, model, step: int):
        """
        Apply freeze/unfreeze to model parameters based on current phase.

        Args:
            model: paddle.nn.Layer with VHA attention layers
            step: current training step
        """
        phase = self.get_phase(step)
        frozen_patterns = self.get_frozen_params(step)

        for name, param in model.named_parameters():
            should_freeze = any(pat in name for pat in frozen_patterns)
            param.stop_gradient = should_freeze
