"""TBO Dataset for PointNeXt training.

Loads .tbo files from a directory and provides PyTorch Dataset interface.
Data is already preprocessed (centered, scaled, rotated) so no transforms needed.
"""
import os
import struct
import glob
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS


@DATASETS.register_module()
class TBODataset(Dataset):
    """Dataset that loads point cloud data from .tbo files.

    TBO v2 format:
        Header: magic(4s) + version(u32) + flags(u32) + asset_count(u32) + channel_count(u32)
        Channel names: null-terminated strings
        UUIDs: asset_count * 16 bytes
        Offsets: asset_count * 4 bytes (cumulative byte offsets)
        Data: interleaved float32 vertex data

    Args:
        data_dir: Directory containing .tbo files
        split: 'train' or 'test' (uses split file or ratio)
        num_points: Number of points per sample (default: 1024)
        in_channels: Number of input channels (default: auto-detect from first file)
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

        # Find all TBO files
        tbo_files = sorted(glob.glob(os.path.join(data_dir, '*.tbo')))
        if not tbo_files:
            raise FileNotFoundError(f'No .tbo files found in {data_dir}')
        logging.info(f'Found {len(tbo_files)} TBO files in {data_dir}')

        # Load all assets from all files
        self.positions = []  # List of (N, 3) arrays
        self.features = []   # List of (N, C) arrays
        self.uuids = []      # List of UUID hex strings

        detected_channels = None
        for tbo_path in tbo_files:
            pos_list, feat_list, uuid_list, ch_count = self._load_tbo_file(tbo_path)
            self.positions.extend(pos_list)
            self.features.extend(feat_list)
            self.uuids.extend(uuid_list)
            if detected_channels is None:
                detected_channels = ch_count
                logging.info(f'Detected {ch_count} channels from {os.path.basename(tbo_path)}')

        # Set in_channels
        if in_channels is None:
            in_channels = detected_channels
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

    def _load_tbo_file(self, path):
        """Load all assets from a single TBO file."""
        file_size = os.path.getsize(path)

        with open(path, 'rb') as f:
            header = f.read(20)
            magic, version, flags, asset_count, channel_count = struct.unpack_from(
                '<4sIIII', header, 0
            )
            if magic.rstrip(b'\x00') != b'TBO':
                raise ValueError(f'Invalid TBO magic in {path}')

            # Read channel names
            for _ in range(channel_count):
                while True:
                    byte = f.read(1)
                    if not byte or byte == b'\x00':
                        break

            # Read UUIDs
            uuid_data = f.read(asset_count * 16)

            # Read cumulative offsets
            offset_data = f.read(asset_count * 4)
            offsets = struct.unpack_from(f'<{asset_count}I', offset_data, 0)

        # Memory-map data section
        header_size = offsets[0] if offsets else 20
        mmap = np.memmap(
            path, dtype=np.float32, mode='r',
            offset=header_size,
            shape=((file_size - header_size) // 4),
        )

        pos_list = []
        feat_list = []
        uuid_list = []

        for i in range(asset_count):
            uuid_hex = uuid_data[i * 16 : (i + 1) * 16].hex()
            offset = (offsets[i] - header_size) // 4
            next_offset = (
                (offsets[i + 1] - header_size) // 4
                if i + 1 < asset_count
                else len(mmap)
            )
            point_count = (next_offset - offset) // channel_count

            if point_count == 0:
                continue

            data = mmap[offset : offset + point_count * channel_count].reshape(
                (point_count, channel_count)
            )
            pos = data[:, 0:3]  # xyz

            # Features: all channels after xyz
            if channel_count > 3:
                feat = data[:, 3:]
            else:
                feat = np.zeros((point_count, 1), dtype=np.float32)

            pos_list.append(pos)
            feat_list.append(feat)
            uuid_list.append(uuid_hex)

        return pos_list, feat_list, uuid_list, channel_count

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
