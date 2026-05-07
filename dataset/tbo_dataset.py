"""TBODataset - Point cloud dataset for TBO (Texture-Based Objects).

Registered with openpoints DATASETS registry for use with build_dataloader_from_cfg.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from ..dataset.build import DATASETS


def _build_x_tensor(pos, feat, in_channels):
    """Build input features tensor from torch tensors.
    
    Pads or truncates feature channels to match in_channels, then
    concatenates with positions.
    """
    if in_channels >= 4:
        needed = in_channels - 3
        if feat.shape[1] < needed:
            pad = torch.zeros(
                feat.shape[0], needed - feat.shape[1],
                dtype=torch.float32, device=feat.device,
            )
            feat = torch.cat([feat, pad], dim=1)
        feat = feat[:, :needed]
        return torch.cat([pos, feat], dim=1)
    # in_channels == 3: pad to 4 channels with constant 1.0 for vec4 alignment
    pad = torch.ones(pos.shape[0], 1, dtype=torch.float32, device=pos.device)
    return torch.cat([pos, pad], dim=1)


@DATASETS.register_module()
class TBODataset(Dataset):
    """PyTorch Dataset for TBO point cloud data.
    
    Stores positions, features, and uuids in memory. Returns dicts with
    'pos', 'x', 'feat', and 'uuids' keys for each sample.
    
    Args:
        positions: list of numpy arrays (N, 3) - point coordinates
        features: list of numpy arrays (N, C) - feature vectors
        uuids: list of string identifiers
        num_points: number of points per sample (all data must be pre-resized)
        in_channels: number of input channels (3 + features)
        transform: optional callable to transform data dict
    """
    
    def __init__(self, positions, features, uuids, num_points=1024, in_channels=None, encoder_indices=None, transform=None):
        if in_channels is None:
            raise ValueError('in_channels is required')
        self.positions = positions
        self.features = features
        self.uuids = uuids
        self.num_points = num_points
        self.in_channels = in_channels
        self.encoder_indices = encoder_indices
        self.transform = transform

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = self.positions[idx].copy()
        feat = self.features[idx].copy()

        # Filter to encoder channels if specified
        if self.encoder_indices is not None:
            feat = feat[:, self.encoder_indices]

        n = len(pos)
        if n != self.num_points:
            raise ValueError(
                f"Sample {self.uuids[idx]} has {n} points but expected {self.num_points}. "
                "All data must already be resized to num_points before passing to the dataset."
            )

        pos = torch.from_numpy(pos)
        feat = torch.from_numpy(feat)

        # Pad feat to 4 channels when in_channels=3 for vec4 alignment
        if self.in_channels == 3 and feat.shape[1] < 4:
            pad = torch.ones(feat.shape[0], 4 - feat.shape[1], dtype=feat.dtype, device=feat.device)
            feat = torch.cat([feat, pad], dim=1)

        data = {
            'pos': pos,
            'x': _build_x_tensor(pos, feat, self.in_channels),
            'feat': feat,
            'uuids': self.uuids[idx],
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    @staticmethod
    def collate_fn(datas):
        """Collate function for TBODataset.
        
        Batches pos, x, feat tensors and keeps uuids as a list.
        """
        pos_list = [d['pos'] for d in datas]
        x_list = [d['x'] for d in datas]
        feat_list = [d['feat'] for d in datas]
        uuids = [d['uuids'] for d in datas]

        pos_batch = torch.stack(pos_list, dim=0)
        x_batch = torch.stack(x_list, dim=0)
        feat_batch = torch.stack(feat_list, dim=0)

        return {
            'pos': pos_batch,
            'x': x_batch,
            'feat': feat_batch,
            'uuids': uuids,
        }
