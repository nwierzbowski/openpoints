"""PointNeXt Encoder-Decoder for Point Cloud Reconstruction.

Feeds full point cloud through PointNextEncoder to produce a compact latent,
then decodes latent + learnable points into a reconstructed point cloud via
feature-weighted Chamfer distance loss with optional feature reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import build_model_from_cfg, MODELS
from ...loss import build_criterion_from_cfg


@MODELS.register_module()
class PointNextMAE(nn.Module):
    """Encoder-decoder with PointNext backbone.

    Architecture:
        1. Apply Gaussian jitter to input points
        2. Encode full point cloud through PointNextEncoder -> latent (B, latent_dim)
        3. Decode latent + learnable points -> reconstructed point cloud
        4. Feature-weighted Chamfer + feature MSE loss
    """

    def __init__(
        self,
        encoder_args,
        latent_dim,
        decoder_points,
        decoder_hidden_dim,
        jitter_sigma,
        jitter_prob,
        channel_indices,
        criterion_args,
        **kwargs,
    ):
        super().__init__()

        # Encoder
        self.encoder = build_model_from_cfg(encoder_args)
        encoder_out_dim = self.encoder.out_channels

        # Pre-computed integer indices for feat tensor column access
        # Set by ApplicationService.validate_channels() — model knows zero channel names
        self.decoder_indices = channel_indices['decoder_indices']  # e.g., [0, 2] for normal+combined
        self.encoder_indices = channel_indices.get('encoder_indices', None)
        self.loss_feature_idx = channel_indices['loss_feature_idx']
        self.num_decoder_features = len(self.decoder_indices)
        self.decoder_out_channels = 3 + self.num_decoder_features

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
            torch.randn(1, decoder_points, self.decoder_out_channels) * 0.1
        )

        # Decoder MLP: [learnable_points (decoder_out_channels) + latent (latent_dim)] -> (decoder_out_channels)
        decoder_head_dim = self.decoder_out_channels + latent_dim
        self.decoder_mlp = nn.Sequential(
            # Stage 1: Massive Unfolding
            nn.Linear(decoder_head_dim, decoder_hidden_dim),
            # nn.LayerNorm(decoder_hidden_dim),
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

            # Stage 5: Multi-Channel Head
            nn.Linear(decoder_hidden_dim // 4, self.decoder_out_channels),
            nn.Tanh(),
        )

        # Jitter augmentation
        self.jitter_sigma = jitter_sigma
        self.jitter_prob = jitter_prob

        # Loss (built from config)
        self.criterion = build_criterion_from_cfg(criterion_args)

    def forward(self, data):
        """Forward pass for training.

        Args:
            data: dict with 'pos' (B, N, 3), 'x' (B, N, C), and 'feat' (B, N, F_raw)

        Returns:
            loss: scalar loss tensor
            pred: reconstructed point cloud (B, decoder_points, decoder_out_channels)
            latent: embedding per sample (B, latent_dim)
        """
        if isinstance(data, dict):
            p = data['pos']  # (B, N, 3)
            f = data.get('x', None)  # (B, N, C) or None
            feat = data.get('feat', None)  # (B, N, F_raw) or None
        else:
            p = data[:, :, :3]
            f = data
            feat = None

        if f is None:
            f = p.transpose(1, 2).contiguous()  # (B, 3, N)
        else:
            f = f.transpose(1, 2).contiguous()  # (B, C, N)

        B, N, _ = p.shape

        # Ensure inputs on same device as model
        device = next(self.parameters()).device
        p = p.to(device)
        f = f.to(device)
        if feat is not None:
            feat = feat.to(device)

        # Keep original feat for loss weighting (before encoder filtering)
        original_feat = feat

        # Filter feat to only encoder channels if configured
        if self.encoder_indices is not None and feat is not None:
            if len(self.encoder_indices) > 0:
                feat = feat[:, :, self.encoder_indices]  # (B, N, num_encoder_features)
            else:
                feat = torch.empty(B, N, 0, device=original_feat.device)

        # Build true_features tensor with only decoder output channels
        if original_feat is not None and self.decoder_indices and len(self.decoder_indices) > 0:
            true_features = original_feat[:, :, self.decoder_indices]  # (B, N, num_decoder_features)
        else:
            true_features = torch.empty(B, N, 0, device=device)

        # Build loss_feature for Chamfer weighting from TBO data
        # Use original_feat (all channels) so weighting feature is available even if encoder/decoder only use XYZ
        if (original_feat is not None and original_feat.shape[2] > 0 and
            self.loss_feature_idx is not None and self.loss_feature_idx < original_feat.shape[2]):
            loss_feature = original_feat[:, :, self.loss_feature_idx]
        else:
            loss_feature = torch.ones(B, N, device=device)

        # Gaussian jitter (Gaussian noise is better for Denoising Autoencoder!)
        if self.training and torch.rand(1).item() < self.jitter_prob:
            jitter = torch.randn_like(p) * self.jitter_sigma  # (B, N, 3)
            p_jittered = p + jitter
            if f is not None and f.shape[1] >= 3:
                f_jittered = f.clone()
                f_jittered[:, :3, :] = f_jittered[:, :3, :] + jitter.transpose(1, 2)  # (B, 3, N)
            else:
                f_jittered = f
        else:
            p_jittered = p
            f_jittered = f

        # Encode full point cloud -> latent
        latent_global = self.encoder.forward_cls_feat(p_jittered, f_jittered)  # (B, C_out)
        latent = self.latent_proj(latent_global)  # (B, latent_dim)

        # Decode: learnable points + latent -> reconstructed
        pred = self.decode(latent)  # (B, decoder_points, decoder_out_channels)

        # Split pred into XYZ and features
        pred_xyz = pred[:, :, :3]  # (B, P, 3)
        pred_features = pred[:, :, 3:]  # (B, P, num_effective)

        # Loss
        loss = self.criterion(pred_xyz, pred_features, p, true_features, loss_feature)

        return loss, pred, latent

    def decode(self, latent):
        """Decode latent to point cloud.

        Args:
            latent: (B, latent_dim)

        Returns:
            pred: (B, decoder_points, decoder_out_channels)
        """
        B = latent.shape[0]

        # Broadcast learnable points to batch
        points = self.learnable_points.expand(B, -1, -1)  # (B, P, 3)

        # Broadcast latent to each point
        latent_expanded = latent.unsqueeze(1).expand(-1, self.decoder_points, -1)  # (B, P, latent_dim)

        # Concat [point_xyz, latent] per point
        inp = torch.cat([points, latent_expanded], dim=2)  # (B, P, 3 + latent_dim)

        # MLP outputs coordinates + features
        pred = self.decoder_mlp(inp)  # (B, P, decoder_out_channels)

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
