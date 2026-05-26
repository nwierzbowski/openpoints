"""PeelerDataset - All-gold soup generation for adaptive object peeler training.

Takes per-mesh TBO data (272 channels: 256 emb + 16 transform per fragment)
and generates soups by mixing K random assets together. Every fragment in
the soup belongs to a real asset. Y encodes same-asset membership for all
assets in the soup.

Registered with openpoints DATASETS registry.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS


@DATASETS.register_module()
class PeelerDataset(Dataset):
    """All-gold soup dataset for peeler training.

    Generates training samples by mixing K random assets together.
    Each sample is a "soup" of N fragments with an N x N ground truth
    matrix Y where Y_ij=1 if fragments i and j belong to the same asset.

    Args:
        all_embeddings: list of numpy arrays, one per asset, shape (N_i, 256)
        all_transforms: list of numpy arrays, one per asset, shape (N_i, 16)
        max_assets_per_soup: int, maximum number of assets to mix per soup (default 10)
        min_assets_per_soup: int, minimum number of assets per soup (default 2)
        seed: int, random seed for reproducibility
    """

    def __init__(
        self,
        all_embeddings,
        all_transforms,
        max_assets_per_soup=10,
        min_assets_per_soup=2,
        max_asset_fragments=None,
        seed=42,
        translation_scale=0.0,
    ):
        self.all_embeddings = all_embeddings  # list of (N_i, 256)
        self.all_transforms = all_transforms  # list of (N_i, 16)
        self.max_assets_per_soup = max_assets_per_soup
        self.min_assets_per_soup = min_assets_per_soup
        self.max_asset_fragments = max_asset_fragments
        self.translation_scale = translation_scale
        self.rng = np.random.RandomState(seed)

        # Subsample oversized assets to cap per-asset fragment count
        if max_asset_fragments is not None:
            subsampled_emb = []
            subsampled_trans = []
            for emb, trans in zip(all_embeddings, all_transforms):
                n = len(emb)
                if n > max_asset_fragments:
                    indices = self.rng.choice(n, size=max_asset_fragments, replace=False)
                    indices = np.sort(indices)
                    subsampled_emb.append(emb[indices])
                    subsampled_trans.append(trans[indices])
                else:
                    subsampled_emb.append(emb)
                    subsampled_trans.append(trans)
            self.all_embeddings = subsampled_emb
            self.all_transforms = subsampled_trans

        self.n_assets = len(self.all_embeddings)

    def __len__(self):
        return self.n_assets

    def __getitem__(self, idx):
        # Sample K assets with random size in [min, max]
        upper = min(self.max_assets_per_soup, self.n_assets)
        lower = min(self.min_assets_per_soup, upper)
        k = self.rng.randint(lower, upper + 1)  # uniform [lower, upper]
        asset_indices = list(self.rng.choice(self.n_assets, size=k, replace=False))

        # Combine all fragments from selected assets
        soup_emb_list = []
        soup_trans_list = []
        asset_ids_list = []
        orig_indices_list = []

        for aid, asset_gid in enumerate(asset_indices):
            emb = self.all_embeddings[asset_gid]
            trans = self.all_transforms[asset_gid]
            soup_emb_list.append(emb)
            soup_trans_list.append(trans)
            asset_ids_list.append(np.full(len(emb), aid, dtype=np.int64))
            # Encode (asset_gid, frag_idx) uniformly
            for fi in range(len(emb)):
                orig_indices_list.append(asset_gid * 100000 + fi)

        soup_emb = np.concatenate(soup_emb_list, axis=0)
        soup_trans = np.concatenate(soup_trans_list, axis=0)
        asset_ids = np.concatenate(asset_ids_list, axis=0)
        orig_indices = np.array(orig_indices_list, dtype=np.int64)

        # Random translation augmentation: per-asset translation in world space
        translation_scale = getattr(self, 'translation_scale', 0.0)
        if translation_scale > 0:
            offset = 0
            for asset_gid in asset_indices:
                n_fragments = len(self.all_transforms[asset_gid])
                t = self.rng.randn(3).astype(np.float32) * translation_scale
                # Translation is at indices 3, 7, 11 of each row (4th column of row-major 4x4)
                soup_trans[offset:offset + n_fragments, 3] += t[0]
                soup_trans[offset:offset + n_fragments, 7] += t[1]
                soup_trans[offset:offset + n_fragments, 11] += t[2]
                offset += n_fragments

        # Shuffle soup
        shuffle_idx = self.rng.permutation(len(soup_emb))
        soup_emb = soup_emb[shuffle_idx]
        soup_trans = soup_trans[shuffle_idx]
        asset_ids = asset_ids[shuffle_idx]
        orig_indices = orig_indices[shuffle_idx]

        # Build N x N ground truth: Y_ij = 1 if same asset
        Y = (asset_ids[:, None] == asset_ids[None, :]).astype(np.float32)

        return {
            'embeddings': torch.from_numpy(soup_emb),
            'transforms': torch.from_numpy(soup_trans),
            'Y': torch.from_numpy(Y),
            'orig_indices': torch.from_numpy(orig_indices),
        }

    @staticmethod
    def collate_fn(datas):
        """Collate function for PeelerDataset.

        Pads samples to the longest soup in the batch.
        """
        max_n = max(len(d['embeddings']) for d in datas)

        embeddings_list = []
        transforms_list = []
        Y_list = []
        orig_indices_list = []
        mask_list = []

        # Pre-create a flat identity matrix for padding [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        identity_flat = torch.eye(4).view(16)

        for d in datas:
            n = len(d['embeddings'])
            pad_size = max_n - n

            mask =  torch.cat([torch.ones(n), torch.zeros(pad_size)])
            mask_list.append(mask)

            if pad_size > 0:
                embeddings_list.append(
                    torch.cat([
                        d['embeddings'],
                        torch.zeros(pad_size, 256, dtype=torch.float32),
                    ], dim=0)
                )
                transforms_list.append(
                    torch.cat([
                        d['transforms'],
                        identity_flat.repeat(pad_size, 1),
                    ], dim=0)
                )
                Y_list.append(
                    torch.cat([
                        torch.cat([d['Y'], torch.zeros(n, pad_size, dtype=torch.float32)], dim=1),
                        torch.zeros(pad_size, max_n, dtype=torch.float32),
                    ], dim=0)
                )
                orig_indices_list.append(
                    torch.cat([
                        d['orig_indices'],
                        torch.full((pad_size,), -999, dtype=torch.int64),
                    ], dim=0)
                )
            else:
                embeddings_list.append(d['embeddings'])
                transforms_list.append(d['transforms'])
                Y_list.append(d['Y'])
                orig_indices_list.append(d['orig_indices'])

        return {
            'embeddings': torch.stack(embeddings_list),
            'transforms': torch.stack(transforms_list),
            'Y': torch.stack(Y_list),
            'orig_indices': torch.stack(orig_indices_list),
            'mask': torch.stack(mask_list), # [B, N] 
        }
