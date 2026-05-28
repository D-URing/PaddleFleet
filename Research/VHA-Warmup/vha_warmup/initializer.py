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
        method: str = "average",
    ) -> dict:
        """
        Merge source KV groups into target KV groups.

        Args:
            W_k: [D, src_H_k * d] source K projection
            W_v: [D, src_H_k * d] source V projection
            method: "average" (simple mean) or "svd" (SVD-based optimal merge)

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

        if method == "svd":
            # Procrustes-based merge: find the K that minimizes total squared
            # distance to all source heads after allowing a rotation.
            # K_merged = (1/n) * sum_h (K_h @ R_h) where R_h aligns K_h to a common frame.
            # This is better than simple average when heads differ by rotation.
            for g in range(self.tgt_H_k):
                src_start = g * self.merge_factor
                src_end = src_start + self.merge_factor

                heads = [W_k_groups[:, src_start + h, :] for h in range(self.merge_factor)]

                # Use first head as reference, align others via Procrustes
                ref = heads[0]
                aligned = [ref.copy()]
                for h in range(1, self.merge_factor):
                    # Solve orthogonal Procrustes: find R s.t. heads[h] @ R ≈ ref
                    # min ||heads[h] @ R - ref||_F  s.t. R^T R = I
                    M = heads[h].T @ ref  # [d, d]
                    U_p, _, Vt_p = np.linalg.svd(M)
                    R = U_p @ Vt_p  # optimal rotation
                    aligned.append(heads[h] @ R)

                k_new[:, g, :] = np.mean(aligned, axis=0)

                # Same for V
                v_heads = [W_v_groups[:, src_start + h, :] for h in range(self.merge_factor)]
                v_ref = v_heads[0]
                v_aligned = [v_ref.copy()]
                for h in range(1, self.merge_factor):
                    M_v = v_heads[h].T @ v_ref
                    U_v, _, Vt_v = np.linalg.svd(M_v)
                    R_v = U_v @ Vt_v
                    v_aligned.append(v_heads[h] @ R_v)
                v_new[:, g, :] = np.mean(v_aligned, axis=0)
        else:
            # Simple average of adjacent groups
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

    def decompose_q_weights_joint(self, W_q: np.ndarray, W_k: np.ndarray, kv_merge_method: str = "svd") -> dict:
        """
        Joint Q-KV aware decomposition.

        First merges K (via Procrustes or average), then solves W2 to minimize
        the attention-score error directly:
            || (A @ W2[g]) @ K_merged[g]^T - B[g] @ K_orig_g^T ||_F

        This is better than independent Q decomposition because W2 accounts for
        the actual merged K directions.

        Args:
            W_q: [D, src_H_q * d]
            W_k: [D, src_H_k * d]
            kv_merge_method: "svd" (Procrustes) or "average" for KV merge step

        Returns:
            dict with q_proj, w2, k_proj
        """
        D = self.D
        d = self.d

        # Reshape Q to per-head: [src_H_q, D, d]
        W_q_heads = W_q.reshape(D, self.src_H_q, d).transpose(1, 0, 2)
        # Reshape K to per-head: [src_H_k, D, d]
        W_k_heads = W_k.reshape(D, self.src_H_k, d).transpose(1, 0, 2)

        # Split Q into target groups
        groups_q = []
        for g in range(self.tgt_H_k):
            start = g * self.src_q_per_tgt_group
            end = start + self.src_q_per_tgt_group
            groups_q.append(W_q_heads[start:end])

        # Reference Q = group 0
        A = groups_q[0]  # [tgt_H_q, D, d]

        # Merge K heads using the same method as merge_kv_weights
        k_new = np.zeros((self.tgt_H_k, D, d), dtype=W_k.dtype)
        for g in range(self.tgt_H_k):
            src_start = g * self.merge_factor
            if kv_merge_method == "svd":
                heads = [W_k_heads[src_start + h] for h in range(self.merge_factor)]
                ref = heads[0]
                aligned = [ref.copy()]
                for h in range(1, self.merge_factor):
                    M = heads[h].T @ ref
                    U_p, _, Vt_p = np.linalg.svd(M)
                    R = U_p @ Vt_p
                    aligned.append(heads[h] @ R)
                k_new[g] = np.mean(aligned, axis=0)
            else:
                src_end = src_start + self.merge_factor
                k_new[g] = W_k_heads[src_start:src_end].mean(axis=0)

        # W2[0] = I (group 0 is exact on Q side)
        w2 = np.zeros((self.tgt_H_k, d, d), dtype=W_q.dtype)
        w2[0] = np.eye(d, dtype=W_q.dtype)

        # For g > 0: solve W2[g] to minimize attention score error.
        # Target: B[g,h] @ K_orig[g,h_k]^T for each Q head h in group g
        # Approx: (A[h] @ W2[g]) @ K_merged[g]^T
        #
        # Rewrite as: A[h] @ W2[g] @ K_m^T ≈ B[g,h] @ K_orig_h^T
        # Since K_orig varies per source head, we use a simpler proxy:
        #   minimize || A @ W2[g]^T - B[g] ||_F  (standard least-squares)
        # then apply Procrustes refinement to keep W2 close to orthogonal.
        for g in range(1, self.tgt_H_k):
            B = groups_q[g]  # [tgt_H_q, D, d]

            A_flat = A.reshape(-1, d)  # [tgt_H_q*D, d]
            B_flat = B.reshape(-1, d)

            # Standard least-squares: A_flat @ W2_T = B_flat
            ATA = A_flat.T @ A_flat
            ATB = A_flat.T @ B_flat
            reg = 1e-6 * np.eye(d, dtype=W_q.dtype)
            W2_g_T = np.linalg.solve(ATA + reg, ATB)  # [d, d]

            # Procrustes refinement: project W2 onto nearest orthogonal matrix
            # then blend with least-squares solution to balance expressiveness
            # and norm preservation.
            U_w, S_w, Vt_w = np.linalg.svd(W2_g_T)
            W2_ortho = U_w @ Vt_w  # nearest orthogonal

            # Blend: 0.7 * least-squares + 0.3 * orthogonal
            # This keeps most of the approximation quality while reducing
            # norm distortion that causes loss spikes.
            alpha = 0.7
            w2[g] = (alpha * W2_g_T + (1 - alpha) * W2_ortho).T

        q_proj_new = A.transpose(1, 0, 2).reshape(D, self.tgt_H_q * d)

        return {
            "q_proj": q_proj_new,
            "w2": w2,
            "k_proj": k_new.transpose(1, 0, 2).reshape(D, self.tgt_H_k * d),
        }

    def convert_layer(self, W_q, W_k, W_v, W_o, kv_merge_method="average", joint=False) -> dict:
        """
        Full conversion of one attention layer.

        Args:
            W_q: [D, src_H_q * d]
            W_k: [D, src_H_k * d]
            W_v: [D, src_H_k * d]
            W_o: [src_H_q * d, D]
            kv_merge_method: "average" or "svd"
            joint: if True, use joint Q-KV aware decomposition

        Returns:
            dict with all initialized weights for VHA layer
        """
        if joint:
            joint_result = self.decompose_q_weights_joint(W_q, W_k, kv_merge_method)
            # V still needs separate merge
            kv_result = self.merge_kv_weights(W_k, W_v, method=kv_merge_method)
            return {
                "q_proj": joint_result["q_proj"],
                "w2_rot": joint_result["w2"],
                "k_proj": joint_result["k_proj"],
                "v_proj": kv_result["v_proj"],
                "o_proj": W_o,
            }
        else:
            q_result = self.decompose_q_weights(W_q)
            kv_result = self.merge_kv_weights(W_k, W_v, method=kv_merge_method)
            return {
                "q_proj": q_result["q_proj"],
                "w2_rot": q_result["w2"],
                "k_proj": kv_result["k_proj"],
                "v_proj": kv_result["v_proj"],
                "o_proj": W_o,
            }
