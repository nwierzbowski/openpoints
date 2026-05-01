"""TBO Dataset for PointNeXt training.

Accesses TBO data already loaded by TBOManager (via Training Import tab).
Data is already preprocessed (centered, scaled, rotated) so no transforms needed.
"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS


@DATASETS.register_module()
class TBODataset(Dataset):
    """Dataset that accesses point cloud data from TBOManager.

    Args:
        data_dir: Directory containing .tbo files (used to verify files exist)
        split: 'train' or 'test' (uses split file or ratio)
        num_points: Number of points per sample (default: 1024)
        in_channels: Number of input channels (default: auto-detect from TBOManager)
        split_file: Optional file with UUIDs for train/test assignment
        split_ratio: Fraction for train split (default: 0.8)
    """

    def __init__(
        self,
        data_dir,
        split='train',
        num_points=1024,
        in_channels=None,
        split_file=None,
        split_ratio=0.8,
        transform=None,
        **kwargs,
    ):
        super().__init__()
        self.partition = split
        self.num_points = num_points
        self.transform = transform
        self.data_dir = data_dir

        # Access assets already loaded by TBOManager (via Training Import tab)
        from curator.application.services.tbo_manager import TBOManager
        manager = TBOManager.instance()
        if not manager.is_loaded:
            raise RuntimeError('TBOManager has no loaded data. Load files via Training Import tab first.')

        self.positions = [a.positions for a in manager.assets if a.split == self.partition]
        self.features = [a.features for a in manager.assets if a.split == self.partition]
        self.uuids = [a.uuid for a in manager.assets if a.split == self.partition]

        # Set in_channels from TBOManager
        if in_channels is None:
            ch_count = manager.channel_count
            if ch_count is None:
                raise RuntimeError('Cannot detect channel count from TBOManager')
            in_channels = ch_count
            logging.info(f'Detected {in_channels} channels from TBOManager')
        self.in_channels = in_channels

        # Apply train/test split
        if split_file:
            self._apply_split_file(split_file)
        else:
            self._apply_random_split(split_ratio)

        logging.info(
            f'TBODataset: {len(self)} samples for {split} split, '
            f'{self.num_points} points, {self.in_channels} channels'
        )

    def _apply_split_file(self, split_file):
        """Apply split based on UUID file."""
        with open(split_file, 'r') as f:
            split_uuids = set(line.strip() for line in f if line.strip())

        mask = [uuid in split_uuids for uuid in self.uuids]
        self.positions = [p for p, m in zip(self.positions, mask) if m]
        self.features = [f for f, m in zip(self.features, mask) if m]
        self.uuids = [u for u, m in zip(self.uuids, mask) if m]

    def _apply_random_split(self, split_ratio):
        """Apply random train/test split."""
        n = len(self.positions)
        indices = np.random.RandomState(42).permutation(n)

        if self.partition == 'train':
            split_idx = int(n * split_ratio)
            selected = indices[:split_idx]
        else:
            split_idx = int(n * split_ratio)
            selected = indices[split_idx:]

        self.positions = [self.positions[i] for i in selected]
        self.features = [self.features[i] for i in selected]
        self.uuids = [self.uuids[i] for i in selected]

    def __getitem__(self, idx):
        pos = self.positions[idx].astype(np.float32)
        feat = self.features[idx].astype(np.float32)

        # Subsample to num_points if needed
        if len(pos) > self.num_points:
            perm = np.random.choice(len(pos), self.num_points, replace=False)
            pos = pos[perm]
            feat = feat[perm]
        elif len(pos) < self.num_points:
            # Pad with zeros
            pad_size = self.num_points - len(pos)
            pos = np.pad(pos, ((0, pad_size), (0, 0)), constant_values=0)
            feat = np.pad(feat, ((0, pad_size), (0, 0)), constant_values=0)

        # Ensure feature channels match in_channels
        if self.in_channels > 3:
            needed_feat_channels = self.in_channels - 3
            if feat.shape[1] < needed_feat_channels:
                feat = np.pad(
                    feat, ((0, 0), (0, needed_feat_channels - feat.shape[1])),
                    constant_values=0
                )
            feat = feat[:, :needed_feat_channels]
            x = np.concatenate([pos, feat], axis=1)
        else:
            x = pos

        data = {
            'pos': torch.from_numpy(pos),       # (N, 3)
            'x': torch.from_numpy(x),            # (N, in_channels)
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.positions)
