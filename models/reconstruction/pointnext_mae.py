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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # This math is blistering fast in BF16
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight
    
class GeGLU(nn.Module):
    """
    ONNX-safe Gated Linear Unit with GELU activation.
    Splits the last dimension in half and uses one half to gate the other.
    """
    def forward(self, x):
        # chunk(2, dim=-1) splits the channel dimension into two equal tensors
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.nn.functional.silu(x2)

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, query_dim):
        super().__init__()

        width = query_dim + latent_dim
        
        self.layer1 = nn.Sequential(
            nn.Linear(width, latent_dim * 2),
            GeGLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            GeGLU(),
        )
        
        # Stage 3: [Stage2_Out + Latent] -> Hidden/2 (e.g. 1280 -> 512)
        self.layer3 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            GeGLU(),
        )
        
        # # Stage 4: [Stage3_Out + Latent] -> Hidden/4 (e.g. 768 -> 256)
        self.layer4 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            GeGLU(),
        )

        self.layer5 = nn.Sequential(nn.Linear(latent_dim, query_dim * 2), GeGLU())

    def forward(self, queries, latent):
        # latent is already (B, P, latent_dim) from the decoder
        
        x = latent

        x = x + self.layer1(torch.cat([queries, latent], dim=-1))

        x = x + self.layer2(x)
        
        x = x + self.layer3(x)

        x = x + self.layer4(x)

        x = self.layer5(x)
        
        return x


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
        # Set by channel validation — model knows zero channel names
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
            torch.randn(1, decoder_points, self.decoder_out_channels) * 0.4
        )

        # Decoder: ConditionalDecoder with latent injected at every layer
        self.decoder = ConditionalDecoder(
            latent_dim=latent_dim,
            query_dim=self.decoder_out_channels,
        )
        # Zero-init final layer so we start at the random initialization
        # nn.init.constant_(self.decoder.layer5.weight, 0)
        # nn.init.constant_(self.decoder.layer5.bias, 0)

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

        B, N, _ = p.shape
        device = p.device
        
        p_jittered = p
        f_jittered = f

        if self.training:
            # 1. Create a mask for each batch element (B, 1, 1)
            # This determines which samples in the batch get jittered
            mask = (torch.rand(B, 1, 1, device=device) < self.jitter_prob).to(p.dtype)
            
            # 2. Generate jitter for the whole batch
            # Multiplying by mask zeros out jitter for samples that didn't "pass" the prob check
            jitter = torch.randn_like(p) * self.jitter_sigma * mask  # (B, N, 3)
            
            # 3. Apply jitter to p
            p_jittered = p + jitter
            
            # 4. Apply jitter to f (if applicable)
            if f is not None and f.shape[1] >= 3:
                f_jittered = f.clone()
                # f is usually (B, 3, N), so we transpose jitter from (B, N, 3) to (B, 3, N)
                f_jittered[:, :3, :] += jitter.transpose(1, 2)

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
        points = self.learnable_points.expand(B, -1, -1)  # (B, P, decoder_out_channels)

        # Broadcast latent to each point
        latent_expanded = latent.unsqueeze(1).expand(-1, self.decoder_points, -1)  # (B, P, latent_dim)

        # Decoder with latent injected at every layer
        pred = self.decoder(points, latent_expanded)  # (B, P, decoder_out_channels)

        reconstructed_points = points + pred

        return reconstructed_points

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
