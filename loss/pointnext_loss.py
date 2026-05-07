"""Feature-weighted Chamfer distance + feature MSE loss for PointNeXt MAE.

Registered with openpoints LOSS registry for use with build_criterion_from_cfg.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import chamfer
from .build import LOSS


class ChamferWeightedFunction(torch.autograd.Function):
    """Chamfer distance with indices, preserving autograd graph.

    Mirrors ChamferFunction but returns indices for feature matching.
    Handles AMP by casting to float32 via @custom_fwd.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(
            xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
        )
        return grad_xyz1, grad_xyz2, None, None


@LOSS.register_module()
class FeatureWeightedChamfer(nn.Module):
    """Feature-weighted Chamfer + feature MSE loss.

    Uses spatial nearest-neighbor matching in XYZ space, then computes:
    - Weighted forward Chamfer: mean(||pred-true|| * (1 + alpha * feature))
    - Weighted backward Chamfer: mean(||true-pred|| * (1 + alpha * feature))
    - Feature MSE: feature_weight * MSE(pred_features, true_features_at_matches)

    Args:
        feature_column: Name of TBO feature column to use for Chamfer weighting
        feature_alpha: Multiplier for feature values in Chamfer weighting
        feature_weight: Multiplier for feature MSE component
    """

    def __init__(self, feature_column='Normal Variance', feature_alpha=1.0, feature_weight=1.0):
        super().__init__()
        self.feature_column = feature_column
        self.alpha = feature_alpha
        self.feature_weight = feature_weight

    def forward(self, pred_xyz, pred_features, true_xyz, true_features, loss_feature):
        """Compute feature-weighted loss.

        Args:
            pred_xyz: (B, P, 3) predicted point coordinates
            pred_features: (B, P, F) predicted feature values (decoder output channels)
            true_xyz: (B, N, 3) ground truth point coordinates
            true_features: (B, N, F) ground truth feature values (decoder output channels)
            loss_feature: (B, N) feature values used for Chamfer weighting

        Returns:
            total_loss: scalar tensor
        """
        # Spatial matching: pred -> true (forward) and true -> pred (backward)
        dist1, dist2, idx1, idx2 = ChamferWeightedFunction.apply(pred_xyz, true_xyz)
        # chamfer CUDA extension returns int32 indices; gather() requires int64
        idx1 = idx1.to(torch.int64)
        idx2 = idx2.to(torch.int64)

        # Forward Chamfer (pred->true), weighted by feature values
        if self.alpha > 0:
            true_feature_at_pred = loss_feature.gather(1, idx1)  # (B, P)
            weights = 1.0 + self.alpha * true_feature_at_pred
            forward_chamfer = torch.mean(torch.sqrt(dist1 + 1e-7) * weights)
        else:
            forward_chamfer = torch.mean(torch.sqrt(dist1 + 1e-7))

        # Backward Chamfer (true->pred), weighted by feature values at true points
        if self.alpha > 0:
            backward_chamfer = torch.mean(torch.sqrt(dist2 + 1e-7) * (1.0 + self.alpha * loss_feature))
        else:
            backward_chamfer = torch.mean(torch.sqrt(dist2 + 1e-7))

        # Feature MSE (only if decoder outputs features)
        if pred_features.numel() > 0 and true_features.numel() > 0:
            # Forward: pred_features (B,P,F) vs true_features_at_pred (B,P,F)
            idx_exp_fwd = idx1.long().unsqueeze(-1).expand(-1, -1, pred_features.shape[-1])
            true_features_at_pred = true_features.gather(1, idx_exp_fwd)
            forward_feature_mse = F.mse_loss(pred_features, true_features_at_pred)

            # Backward: true_features (B,N,F) vs pred_features_at_true (B,N,F)
            idx_exp_bwd = idx2.long().unsqueeze(-1).expand(-1, -1, true_features.shape[-1])
            pred_features_at_true = pred_features.gather(1, idx_exp_bwd)
            backward_feature_mse = F.mse_loss(true_features, pred_features_at_true)

            feature_mse = (forward_feature_mse + backward_feature_mse) / 2
        else:
            feature_mse = torch.tensor(0.0, device=pred_xyz.device)

        return (forward_chamfer + backward_chamfer) / 2 + self.feature_weight * feature_mse
