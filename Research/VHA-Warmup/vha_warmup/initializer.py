"""
GQA -> VHA weight initializer.

Converts a pretrained GQA model (e.g., 24Q-4KV) into VHA format (12Q-2KV + W2).

Strategy:
- Q side: Direct copy of source Q[0..H_q'-1], W2[0]=I, W2[1] via least-squares
- KV side: Adjacent group averaging (or activation-aware merge)
- O_proj: Direct inheritance
- Postmix: Zero-init (no effect at start)
"""

import numpy as np

try:
    import paddle
except ImportError:
    paddle = None


class VHAInitializer:
    """Initialize VHA weights from a pretrained GQA checkpoint."""

    def __init__(
        self,
        source_num_q_heads: int = 24,
        source_num_kv_heads: int = 4,
        target_num_q_heads: int = 12,
        target_num_kv_heads: int = 2,
        head_dim: int = 128,
        hidden_size: int = 2048,
    ):
        self.src_H_q = source_num_q_heads
        self.src_H_k = source_num_kv_heads
        self.tgt_H_q = target_num_q_heads
        self.tgt_H_k = target_num_kv_heads
        self.d = head_dim
        self.D = hidden_size

        # Validate: virtual heads should match source
        assert self.tgt_H_q * self.tgt_H_k == self.src_H_q, (
            f"Virtual heads mismatch: {self.tgt_H_q}*{self.tgt_H_k} != {self.src_H_q}"
        )
        # Source groups per target group
        self.merge_factor = self.src_H_k // self.tgt_H_k
        # Source Q heads per target group
        self.src_q_per_tgt_group = self.src_H_q // self.tgt_H_k

    def decompose_q_weights(self, W_q: np.ndarray) -> dict:
        """
        Decompose source Q projection into physical Q + W2 rotations.

        Args:
            W_q: [D, src_H_q * d] source Q projection weight

        Returns:
            dict with:
                - q_proj: [D, tgt_H_q * d] new Q projection
                - w2: [tgt_H_k, d, d] rotation matrices
        """
        D, _ = W_q.shape
        d = self.d

        # Reshape to per-head: [src_H_q, D, d]
        W_q_heads = W_q.reshape(D, self.src_H_q, d).transpose(1, 0, 2)  # [src_H_q, D, d]

        # Split into target groups
        # Group 0: source Q[0 : src_q_per_tgt_group]
        # Group 1: source Q[src_q_per_tgt_group : 2*src_q_per_tgt_group]
        groups = []
        for g in range(self.tgt_H_k):
            start = g * self.src_q_per_tgt_group
            end = start + self.src_q_per_tgt_group
            groups.append(W_q_heads[start:end])  # [src_q_per_tgt_group, D, d]

        # Q_new = Group 0's first tgt_H_q heads (direct copy)
        # Since src_q_per_tgt_group == tgt_H_q, we take all of group 0
        A = groups[0]  # [tgt_H_q, D, d] — reference group

        # W2[0] = Identity (group 0 is exact)
        w2 = np.zeros((self.tgt_H_k, d, d), dtype=W_q.dtype)
        w2[0] = np.eye(d, dtype=W_q.dtype)

        # W2[g] for g > 0: solve least-squares A @ W2[g]^T ≈ B[g]
        for g in range(1, self.tgt_H_k):
            B = groups[g]  # [tgt_H_q, D, d]

            # Stack: A_flat [tgt_H_q*D, d], B_flat [tgt_H_q*D, d]
            A_flat = A.reshape(-1, d)
            B_flat = B.reshape(-1, d)

            # Solve: A_flat @ W2[g]^T = B_flat
            # => W2[g] = (B_flat^T @ A_flat) @ inv(A_flat^T @ A_flat + λI)
            ATA = A_flat.T @ A_flat
            ATB = A_flat.T @ B_flat
            # Regularized solve
            reg = 1e-6 * np.eye(d, dtype=W_q.dtype)
            W2_g_T = np.linalg.solve(ATA + reg, ATB)  # [d, d]
            w2[g] = W2_g_T.T

        # Q_new projection: [D, tgt_H_q * d]
        q_proj_new = A.transpose(1, 0, 2).reshape(D, self.tgt_H_q * d)

        return {"q_proj": q_proj_new, "w2": w2}

    def merge_kv_weights(
        self,
        W_k: np.ndarray,
        W_v: np.ndarray,
    ) -> dict:
        """
        Merge source KV groups into target KV groups.

        Args:
            W_k: [D, src_H_k * d] source K projection
            W_v: [D, src_H_k * d] source V projection

        Returns:
            dict with:
                - k_proj: [D, tgt_H_k * d]
                - v_proj: [D, tgt_H_k * d]
        """
        D = self.D
        d = self.d

        # Reshape to per-group
        W_k_groups = W_k.reshape(D, self.src_H_k, d)  # [D, src_H_k, d]
        W_v_groups = W_v.reshape(D, self.src_H_k, d)

        k_new = np.zeros((D, self.tgt_H_k, d), dtype=W_k.dtype)
        v_new = np.zeros((D, self.tgt_H_k, d), dtype=W_v.dtype)

        # Average adjacent groups
        for g in range(self.tgt_H_k):
            src_start = g * self.merge_factor
            src_end = src_start + self.merge_factor
            k_new[:, g, :] = W_k_groups[:, src_start:src_end, :].mean(axis=1)
            v_new[:, g, :] = W_v_groups[:, src_start:src_end, :].mean(axis=1)

        return {
            "k_proj": k_new.reshape(D, self.tgt_H_k * d),
            "v_proj": v_new.reshape(D, self.tgt_H_k * d),
        }

    def compute_approximation_error(self, W_q: np.ndarray, result: dict) -> dict:
        """
        Compute relative approximation error for Q-side decomposition.

        Returns:
            dict with per-group relative errors
        """
        d = self.d
        D = self.D

        W_q_heads = W_q.reshape(D, self.src_H_q, d).transpose(1, 0, 2)
        q_new_heads = result["q_proj"].reshape(D, self.tgt_H_q, d).transpose(1, 0, 2)
        w2 = result["w2"]

        errors = {}
        for g in range(self.tgt_H_k):
            start = g * self.src_q_per_tgt_group
            end = start + self.src_q_per_tgt_group
            B = W_q_heads[start:end]  # [tgt_H_q, D, d] target

            # Reconstruct: Q_new @ W2[g]^T
            reconstructed = np.einsum("hid,de->hie", q_new_heads, w2[g].T)

            residual = np.linalg.norm(B - reconstructed) ** 2
            total = np.linalg.norm(B) ** 2
            errors[f"group_{g}"] = residual / (total + 1e-12)

        return errors

    def convert_layer(self, W_q, W_k, W_v, W_o) -> dict:
        """
        Full conversion of one attention layer.

        Args:
            W_q: [D, src_H_q * d]
            W_k: [D, src_H_k * d]
            W_v: [D, src_H_k * d]
            W_o: [src_H_q * d, D]

        Returns:
            dict with all initialized weights for VHA layer
        """
        q_result = self.decompose_q_weights(W_q)
        kv_result = self.merge_kv_weights(W_k, W_v)

        return {
            "q_proj": q_result["q_proj"],
            "w2_rot": q_result["w2"],
            "k_proj": kv_result["k_proj"],
            "v_proj": kv_result["v_proj"],
            "o_proj": W_o,  # Direct inheritance (virtual head count preserved)
        }
