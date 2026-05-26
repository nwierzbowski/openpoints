"""Adaptive Object Peeler - single-anchor scene decomposition model.

Architecture:
    1. FeatureLift: [256 DNA + 15 pose] -> 512D
    2. Global Max Pool -> Scene Vector (512) -> concat to each island -> [N, 1024]
    3. AnchorHead: MLP on [N, 1024] -> softmax -> P_anchor distribution
    4. RelationHead: MLP on [512+512 lifted + 15 rel] -> raw logits -> BCE loss

Training: single anchor per batch, BCE between selected anchor's membership
logits and ground truth Y row. No N×N affinity computation.

Registered with openpoints MODELS registry.
"""
import torch
import torch.nn as nn

from ..build import MODELS


class FeatureLift(nn.Module):
    """Normalize raw input features.

    Concatenates 256D embedding + 15D pose features -> 271D,
    then projects to model dimension.
    """

    def __init__(self, embed_dim=256, pose_dim=15, model_dim=512, drop_rate=0.1):
        super().__init__()
        self.input_dim = embed_dim + pose_dim  # 271
        self.model_dim = model_dim
        self.lift = nn.Sequential(
            nn.Linear(self.input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, input_dim) - concatenated embeddings + pose features
        Returns:
            (B, N, model_dim)
        """
        return self.lift(x)


class AnchorHead(nn.Module):
    """MLP that scores each island as a potential anchor seed.

    High scores -> complex, identifiable parts (receiver, barrel).
    Low scores -> simple, redundant parts (screws, noise).
    """

    def __init__(self, model_dim=1024, drop_rate=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(model_dim // 4, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, model_dim) - contextual embeddings [lifted + scene]
        Returns:
            (B, N) - raw anchor logits (softmax applied in Peeler.forward)
        """
        return self.mlp(x).squeeze(-1)


class RelationHead(nn.Module):
    """Deep MLP that scores membership for seed vs candidate pairs.

    Input: [lifted_emb_S, lifted_emb_C, relative_features]
    Output: raw logits (sigmoid applied externally for inference)
    """

    def __init__(self, model_dim=512, rel_feature_dim=15, drop_rate=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.rel_feature_dim = rel_feature_dim
        # Input: seed_emb(512) + cand_emb(512) + rel_features(15) = 1039
        self.input_dim = model_dim * 2 + rel_feature_dim  # 1039
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(model_dim // 4, 1),
        )

    def forward(self, seed_emb, cand_emb, rel_features):
        """
        Args:
            seed_emb: (B, model_dim) - lifted embedding of selected seed
            cand_emb: (B, N, model_dim) - lifted embeddings of all candidates
            rel_features: (B, N, rel_feature_dim) - relative features
        Returns:
            (B, N) - raw membership logits
        """
        seed_expanded = seed_emb.unsqueeze(1).expand(-1, cand_emb.shape[1], -1)
        concat = torch.cat([seed_expanded, cand_emb, rel_features], dim=-1)
        return self.mlp(concat).squeeze(-1)


@MODELS.register_module()
class Peeler(nn.Module):
    """Adaptive Object Peeler model.

    Full forward pass:
        1. Decompose transforms -> pose features
        2. FeatureLift: [emb + pose] -> model_dim
        3. Global Max Pool -> Scene Vector -> concat -> [N, 1024]
        4. AnchorHead: softmax -> P_anchor distribution
        5. RelationHead: raw membership logits for selected seed vs all candidates
    """

    def __init__(
        self,
        embed_dim=256,
        pose_dim=15,
        model_dim=512,
        lift_drop_rate=0.1,
        anchor_drop_rate=0.1,
        relation_drop_rate=0.3,
        **kwargs,
    ):
        super().__init__()

        # Feature lifting
        self.feature_lift = FeatureLift(embed_dim, pose_dim, model_dim, drop_rate=lift_drop_rate)

        # Contextual: global max pool + scene vector concat
        # Anchor head on [N, 1024]
        self.anchor_head = AnchorHead(model_dim * 2, drop_rate=anchor_drop_rate)

        # Relation head on [512+512 lifted + 15 rel]
        self.relation_head = RelationHead(model_dim, pose_dim, drop_rate=relation_drop_rate)

    def decompose_pose(self, transforms):
        """Decompose 4x4 world matrices into pose features.

        Args:
            transforms: (B, N, 16) f32 - row-major 4x4 matrices

        Returns:
            pose_features: (B, N, 15) f32 - [T(3), S(3), R(9)]
        """
        B, N, _ = transforms.shape
        mat = transforms.reshape(B, N, 4, 4)

        translation = mat[:, :, :3, 3]
        scale = torch.norm(mat[:, :, :3, :3], dim=-1)
        rot = mat[:, :, :3, :3] / scale[:, :, None, :]

        translation = (rot.transpose(-2, -1) @ translation.unsqueeze(-1)).squeeze(-1) / scale
        rot_flat = rot.reshape(B, N, -1)

        return torch.cat([translation, scale, rot_flat], dim=-1)

    def compute_relative_features(self, seed_pose, cand_pose):
        """Compute relative features between seed and all candidates.

        Args:
            seed_pose: (B, 15) f32
            cand_pose: (B, N, 15) f32

        Returns:
            rel_features: (B, N, 15) f32
        """
        B = seed_pose.shape[0]
        seed_T = seed_pose[:, :3]
        seed_S = seed_pose[:, 3:6]
        seed_R = seed_pose[:, 6:].reshape(B, 3, 3)

        cand_T = cand_pose[:, :, :3]
        cand_S = cand_pose[:, :, 3:6]
        cand_R = cand_pose[:, :, 6:].reshape(B, -1, 3, 3)

        # Relative position in world space: T_c - T_s
        rel_pos = cand_T - seed_T.unsqueeze(1)

        # Relative scale (log): log10(S_c / S_s)
        rel_scale = torch.log10(cand_S / (seed_S.unsqueeze(1) + 1e-8))

        # Relative rotation: R_s^T * R_c, flattened
        seed_Rt = seed_R.transpose(1, 2).unsqueeze(1)
        rel_rot = torch.matmul(seed_Rt, cand_R)
        rel_rot = rel_rot.reshape(B, -1, 9)

        return torch.cat([rel_pos, rel_scale, rel_rot], dim=-1)

    def forward(self, embeddings, transforms, mask):
        """Forward pass (single anchor training).

        Args:
            embeddings: (B, N, 256) - fragment embeddings
            transforms: (B, N, 16) - fragment transforms
            mask: (B, N) - 1 for real fragments

        Returns:
            anchor_probs: (B, N) - softmax distribution
            membership_logits: (B, N) - raw relation head logits
            lifted_emb: (B, N, model_dim) - lifted embeddings for relation head
            seed_idx: (B,) - selected seed index
        """
        B, N, _ = embeddings.shape

        # Mask: 1 for real fragments, 0 for zero-padded ghosts
        # mask = (transforms.abs().sum(dim=-1) > 0).float()  # (B, N)

        # Decompose transforms -> pose features
        pose_features = self.decompose_pose(transforms)
        # pose_features = torch.nan_to_num(pose_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature lift: [emb + pose] -> model_dim
        raw_input = torch.cat([embeddings, pose_features], dim=-1)
        lifted_emb = self.feature_lift(raw_input)

        mask_expand = mask.unsqueeze(-1)
        masked_for_pool = lifted_emb + (1 - mask_expand) * -1e9

        # Global max pool -> scene vector
        scene_vec = masked_for_pool.max(dim=1, keepdim=True).values  # (B, 1, 512)

        # Context injection: concat scene vector to each island
        context = torch.cat([lifted_emb, scene_vec.expand(-1, N, -1)], dim=-1)  # (B, N, 1024)

        # Anchor head: softmax -> P_anchor distribution
        anchor_logits = self.anchor_head(context)  # (B, N)
        anchor_logits = anchor_logits + (1 - mask) * -1e9  # mask padding before softmax
        anchor_probs = torch.softmax(anchor_logits, dim=1)  # (B, N)

        # Select seed: highest anchor probability
        seed_idx = torch.argmax(anchor_probs, dim=-1, keepdim=True)  # (B, 1)

        # Extract seed pose
        seed_pose = torch.gather(
            pose_features,
            dim=1,
            index=seed_idx.unsqueeze(-1).expand(-1, -1, pose_features.shape[-1]),
        ).squeeze(1)  # (B, 15)

        # Compute relative features: seed vs all candidates
        rel_features = self.compute_relative_features(seed_pose, pose_features)

        # Extract seed lifted embedding
        seed_emb = torch.gather(
            lifted_emb,
            dim=1,
            index=seed_idx.unsqueeze(-1).expand(-1, -1, lifted_emb.shape[-1]),
        ).squeeze(1)  # (B, model_dim)

        # Relation head: raw membership logits
        membership_logits = self.relation_head(seed_emb, lifted_emb, rel_features)

        return anchor_probs, membership_logits, lifted_emb, seed_idx.squeeze(-1)
