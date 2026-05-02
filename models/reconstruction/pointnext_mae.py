"""PointNeXt Encoder-Decoder for Point Cloud Reconstruction.

Feeds full point cloud through PointNeXt encoder to produce a compact latent,
then decodes latent + learnable points into a reconstructed point cloud via
Chamfer distance loss.
"""
import torch
import torch.nn as nn
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from ..build import build_model_from_cfg, MODELS


@MODELS.register_module()
class PointNextMAE(nn.Module):
    """Encoder-decoder with PointNext backbone.

    Architecture:
        1. Apply Gaussian jitter to input points
        2. Encode full point cloud through PointNextEncoder -> latent (B, latent_dim)
        3. Decode latent + learnable points -> reconstructed point cloud
        4. Chamfer loss between predicted and original points
    """

    def __init__(
        self,
        encoder_args,
        latent_dim=256,
        decoder_points=1024,
        decoder_hidden_dim=1024,
        jitter_sigma=0.01,
        jitter_prob=0.9,
        **kwargs,
    ):
        super().__init__()

        # Encoder
        self.encoder = build_model_from_cfg(encoder_args)
        encoder_out_dim = self.encoder.out_channels

        # Project encoder output to latent
        self.latent_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder: learnable points + latent -> reconstructed point cloud
        self.latent_dim = latent_dim
        self.decoder_points = decoder_points

        # Learnable random initialization
        self.learnable_points = nn.Parameter(
            torch.randn(1, decoder_points, 3) * 0.1
        )

        # Decoder MLP: [point_xyz (3) + latent (latent_dim)] -> (3)
        self.decoder_mlp = nn.Sequential(
            # Stage 1: Massive Unfolding
            nn.Linear(3 + latent_dim, decoder_hidden_dim), 
            nn.LayerNorm(1024),
            nn.ReLU(),

            # Stage 2: Structural Logic
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),

            # Stage 3: Feature Refinement
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2),
            nn.ReLU(),

            # Stage 4: High-Frequency Detail
            nn.Linear(decoder_hidden_dim // 2, decoder_hidden_dim // 4),
            nn.ReLU(),

            # Stage 5: The 5-Channel Head
            nn.Linear(decoder_hidden_dim // 4, 3), 
            nn.Tanh(),
        )

        # Jitter augmentation
        self.jitter_sigma = jitter_sigma
        self.jitter_prob = jitter_prob

        # Loss
        self.criterion = ChamferDistanceL1()

    def forward(self, data):
        """Forward pass for training.

        Args:
            data: dict with 'pos' (B, N, 3) and optionally 'x' (B, N, C)

        Returns:
            loss: Chamfer distance scalar
            pred: reconstructed point cloud (B, decoder_points, 3)
            latent: embedding per sample (B, latent_dim)
        """
        if isinstance(data, dict):
            p = data['pos']  # (B, N, 3)
            f = data.get('x', None)  # (B, N, C) or None
        else:
            p = data[:, :, :3]
            f = data

        if f is None:
            f = p.transpose(1, 2).contiguous()  # (B, 3, N)
        else:
            f = f.transpose(1, 2).contiguous()  # (B, C, N)

        B, N, _ = p.shape

        # Ensure inputs on same device as model
        device = next(self.parameters()).device
        p = p.to(device)
        f = f.to(device)

        # Gaussian jitter (Gaussian noise is better for Denoising Autoencoder!)
        if self.training and torch.rand(1).item() < self.jitter_prob:
            jitter = torch.randn_like(p) * self.jitter_sigma  # (B, N, 3)
            p_jittered = p + jitter
            if f is not None and f.shape[1] >= 3:
                f_jittered = f.clone()
                f_jittered[:, :3, :] = f_jittered[:, :3, :] + jitter.transpose(1, 2)  # (B, 3, N)
            else:
                f_jittered = f  # encoder will derive from p_jittered if None
        else:
            p_jittered = p
            f_jittered = f

        # Encode full point cloud -> latent
        latent_global = self.encoder.forward_cls_feat(p_jittered, f_jittered)  # (B, C_out)
        latent = self.latent_proj(latent_global)  # (B, latent_dim)

        # Decode: learnable points + latent -> reconstructed
        pred = self.decode(latent)  # (B, decoder_points, 3)

        # Chamfer loss
        loss = self.criterion(pred, p)

        return loss, pred, latent

    def decode(self, latent):
        """Decode latent to point cloud.

        Args:
            latent: (B, latent_dim)

        Returns:
            pred: (B, decoder_points, 3)
        """
        B = latent.shape[0]

        # Broadcast learnable points to batch
        points = self.learnable_points.expand(B, -1, -1)  # (B, P, 3)

        # Broadcast latent to each point
        latent_expanded = latent.unsqueeze(1).expand(-1, self.decoder_points, -1)  # (B, P, latent_dim)

        # Concat [point_xyz, latent] per point
        inp = torch.cat([points, latent_expanded], dim=2)  # (B, P, 3 + latent_dim)

        # MLP outputs absolute coordinates
        pred = self.decoder_mlp(inp)  # (B, P, 3)

        return pred

    @torch.no_grad()
    def get_latent(self, data):
        """Extract latent embedding without reconstruction.

        Args:
            data: dict with 'pos' (B, N, 3) and optionally 'x' (B, N, C)

        Returns:
            latent: (B, latent_dim) embedding tensor
        """
        if isinstance(data, dict):
            p = data['pos']
            f = data.get('x', None)
        else:
            p = data[:, :, :3]
            f = data

        if f is None:
            f = p.transpose(1, 2).contiguous()
        else:
            f = f.transpose(1, 2).contiguous()

        device = next(self.parameters()).device
        p = p.to(device)
        f = f.to(device)

        latent_global = self.encoder.forward_cls_feat(p, f)
        latent = self.latent_proj(latent_global)

        return latent
