# group layer: spatial grouping for Morton-ordered points
# 
# Replaces ballquery/KNN with stride-based spatial indexing.
# Points are pre-sorted by 3D Morton code in pivot-engine,
# so spatial neighbors are contiguous in the array.

from typing import Tuple
import copy
import torch
import torch.nn as nn


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch gather (no custom backward).

    Used as ONNX export fallback — ONNX may fail to trace through
    the autograd Function wrapper in GroupingOp.

    Args:
        features: (B, C, N) tensor of features
        idx: (B, npoint, nsample) tensor containing the indices of features to group with

    Returns:
        output: (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1).unsqueeze(1)  # (B, 1, npoint*nsample)
    all_idx = all_idx.expand(-1, features.shape[1], -1)   # (B, C, npoint*nsample) — zero-copy view
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


def torch_grouping_operation(features, idx):
    """Alias for grouping_operation (from torch points kernels)."""
    return grouping_operation(features, idx)


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather features by index using torch.gather.
    
    Pure PyTorch implementation (replaces CUDA pointnet2_cuda.gather_points_wrapper).
    
    Args:
        features: (B, C, N) tensor of features
        idx: (B, npoint) index tensor
    
    Returns:
        output: (B, C, npoint) gathered features
    """
    B, npoint = idx.size()
    _, C, N = features.size()
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    return features.gather(2, idx_expanded)


class GroupAll(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


def spatial_group(query_xyz, support_xyz, features, stride, nsample):
    """Group features using spatial ordering (Morton-coded points).
    
    For stride-based subsampling, neighbors of query point i in the
    downsampled set correspond to a window around index (i * stride)
    in the support set. No distance computation needed.
    
    Args:
        query_xyz: (B, npoint, 3) — downsampled points
        support_xyz: (B, N, 3) — full set (Morton-ordered)
        features: (B, C, N) — feature tensor
        stride: N // npoint — downsampling ratio
        nsample: number of neighbors to gather
    
    Returns:
        dp: (B, 3, npoint, nsample) — relative positions
        fj: (B, C, npoint, nsample) — grouped features
    """
    B, npoint, _ = query_xyz.shape
    N = support_xyz.shape[1]
    half = nsample // 2

    # Center indices: [0, stride, 2*stride, ..., (npoint-1)*stride]
    # Shape: (npoint,)
    center = torch.arange(npoint, device=query_xyz.device) * stride
    # Window: [center-half, ..., center+half-1] for even nsample
    # For nsample=32: offsets = [-16, -15, ..., 0, ..., 15] (32 values)
    offsets = torch.arange(-half, half, device=query_xyz.device)
    idx = (center.unsqueeze(-1) + offsets.unsqueeze(0)).clamp(0, N - 1)
    # Expand to batch: (B, npoint, nsample)
    idx = idx.unsqueeze(0).expand(B, -1, -1)

    # Group positions using grouping_operation (supports (B, 3, N) -> (B, 3, npoint, nsample))
    xyz_trans = support_xyz.transpose(1, 2)  # (B, 3, N)
    grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
    # Relative positions
    query_xyz_t = query_xyz.transpose(1, 2).unsqueeze(-1)  # (B, 3, npoint, 1)
    dp = grouped_xyz - query_xyz_t

    # Group features using grouping_operation
    fj = grouping_operation(features, idx)

    return dp, fj


def spatial_self_group(p, f, nsample):
    """Self-grouping: neighbors of each point in the same set.
    
    For Morton-ordered points, spatial neighbors are just the window
    around each index. Used by LocalAggregation (InvResMLP blocks).
    
    Args:
        p: (B, N, 3) positions
        f: (B, C, N) features
        nsample: number of neighbors
    
    Returns:
        dp: (B, 3, N, nsample) — relative positions
        fj: (B, C, N, nsample) — grouped features
    """
    B, N, _ = p.shape
    half = nsample // 2

    # Window indices: [i-half, ..., i+half-1] for even nsample
    # For nsample=32: offsets = [-16, -15, ..., 0, ..., 15] (32 values)
    idx = (torch.arange(N, device=p.device).unsqueeze(-1) +
           torch.arange(-half, half, device=p.device).unsqueeze(0)).clamp(0, N - 1)
    # Expand to batch: (B, N, nsample)
    idx = idx.unsqueeze(0).expand(B, -1, -1)

    # Group positions using grouping_operation
    p_trans = p.transpose(1, 2)  # (B, 3, N)
    grouped_p = grouping_operation(p_trans, idx)  # (B, 3, N, nsample)
    # Relative positions
    p_t = p.transpose(1, 2).unsqueeze(-1)  # (B, 3, N, 1)
    dp = grouped_p - p_t

    # Group features using grouping_operation
    fj = grouping_operation(f, idx)

    return dp, fj


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


# ============================================================================
# Deprecated stubs — kept for backward compatibility with other PointNeXt models.
# These classes are NOT used by PointNextMAE/PointNextEncoder in pivot-engine.
# pivot-engine uses spatial grouping (spatial_group, spatial_self_group) instead.
# ============================================================================

class KNN(nn.Module):
    """Deprecated — removed. pivot-engine uses spatial grouping."""
    def __init__(self, neighbors, transpose_mode=True):
        super().__init__()
        self.neighbors = neighbors
    def forward(self, support, query):
        raise NotImplementedError('KNN removed. pivot-engine uses spatial grouping.')


class DenseDilated(nn.Module):
    """Deprecated — removed."""
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super().__init__()
        self.dilation = dilation
    def forward(self, edge_index):
        raise NotImplementedError('DenseDilated removed.')


class DilatedKNN(nn.Module):
    """Deprecated — removed."""
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super().__init__()
    def forward(self, query):
        raise NotImplementedError('DilatedKNN removed.')


class BallQuery(nn.Module):
    """Deprecated — removed. pivot-engine uses spatial grouping."""
    def __init__(self, radius: float, nsample: int):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor):
        raise NotImplementedError('BallQuery removed. pivot-engine uses spatial grouping.')


def ball_query(radius: float, nsample: int, support_xyz: torch.Tensor, query_xyz: torch.Tensor):
    """Deprecated — removed."""
    raise NotImplementedError('ball_query removed. pivot-engine uses spatial grouping.')


class QueryAndGroup(nn.Module):
    """Deprecated — removed. pivot-engine uses spatial grouping."""
    def __init__(self, radius: float, nsample: int, **kwargs):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
    def forward(self, query_xyz, support_xyz, features=None):
        raise NotImplementedError('QueryAndGroup removed. pivot-engine uses spatial grouping.')


class KNNGroup(nn.Module):
    """Deprecated — removed. pivot-engine uses spatial grouping."""
    def __init__(self, nsample: int, **kwargs):
        super().__init__()
        self.nsample = nsample
    def forward(self, query_xyz, support_xyz, features=None):
        raise NotImplementedError('KNNGroup removed. pivot-engine uses spatial grouping.')


def create_grouper(group_args):
    """Backward compatibility wrapper.
    
    All grouping in pivot-engine now uses spatial indexing (Morton-ordered points).
    This wrapper is kept for backward compatibility with other PointNeXt models.
    Returns GroupAll (groups all points) — other models using this will not work
    correctly but will not crash on import.
    """
    return GroupAll()
