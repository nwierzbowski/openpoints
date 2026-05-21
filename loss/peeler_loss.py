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

        # BCE between logits and ground truth: (B, N)
        bce_per_frag = self.bce(membership_logits, Y_selected)

        # Masked mean: only count non-padded fragments
        bce_loss = self.membership_weight * (bce_per_frag * mask).sum() / mask.sum().clamp(min=1.0)

        # Weight by 1/P_anchor for gradient flow to anchor head
        p_anchor = anchor_probs[rows, seed_idx].clamp(min=0.01)
        loss = bce_loss / p_anchor.mean()

        return loss, {
            'loss_total': loss.item(),
            'loss_membership': bce_loss.item(),
        }
