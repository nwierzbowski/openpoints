"""Peeler loss functions.

Single-anchor BCE loss with straight-through gradient for anchor head.

For each batch, the model selects one anchor via argmax. The BCE loss
is weighted by 1/P_anchor so the anchor head learns to assign high
probability to anchors that produce clean slices.
"""
import torch
import torch.nn as nn

from .build import LOSS


@LOSS.register_module()
class PeelerLoss(nn.Module):
    """Single-anchor BCE loss with anchor head gradient flow.

    Loss = BCE(membership_logits, Y_selected) / mean(P_anchor)

    The 1/P_anchor weighting lets gradients flow back through softmax
    to the anchor head, training it to prefer anchors that yield good slices.

    Args:
        membership_weight: float, weight for BCE loss (default 1.0)
    """

    def __init__(
        self,
        membership_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.membership_weight = membership_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, membership_logits, Y, anchor_probs, seed_idx):
        """
        Args:
            membership_logits: (B, N) - raw logits for selected anchor
            Y: (B, N, N) - ground truth same-asset matrix
            anchor_probs: (B, N) - softmax distribution over anchors
            seed_idx: (B,) - selected seed index per batch

        Returns:
            loss: scalar total loss
            loss_dict: dict with individual loss components
        """
        B, N, _ = Y.shape

        # Gather Y row for selected anchor: (B, N)
        rows = torch.arange(B, device=Y.device)
        Y_selected = Y[rows, seed_idx]

        # Mask: real fragments have Y[i,i]=1 (self-membership), padding is all zeros
        mask = torch.diagonal(Y, dim1=1, dim2=2)  # (B, N) - 1 for real, 0 for padding

        # Separate positives (same-asset) and negatives (different-asset)
        pos_mask = Y_selected > 0.5  # (B, N) — True for same-asset pairs
        neg_mask = (Y_selected < 0.5) & (mask > 0.5)  # (B, N) — True for different-asset pairs

        # BCE between logits and ground truth: (B, N)
        bce_all = self.bce(membership_logits, Y_selected)

        # Mean BCE for positives and negatives separately
        pos_loss = (bce_all * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
        neg_loss = (bce_all * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)

        # Balanced loss: equal weight to pos and neg
        bce_loss = (0.5 * pos_loss + 0.5 * neg_loss)

        # Weight by 1/P_anchor for gradient flow to anchor head
        p_anchor = anchor_probs[rows, seed_idx]

        baseline = bce_loss.detach().mean()
        advantage = bce_loss.detach() - baseline

        anchor_loss = (torch.log(p_anchor + 1e-8) * advantage).mean()

        # Force the model to stay slightly 'curious'
        # entropy_loss is low when one p is 1.0 and others are 0.0
        entropy_loss = -(anchor_probs * torch.log(anchor_probs + 1e-8)).sum(dim=1).mean()

        loss = bce_loss + anchor_loss - (0.01 * entropy_loss)

        return loss, {
            'loss_total': loss.item(),
            'loss_membership': bce_loss.item(),
        }
