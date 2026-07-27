# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import os

import torch

try:
    import chamfer
except ImportError:
    # The tensor backend below is intentionally independent of this optional,
    # ABI-sensitive extension.
    chamfer = None


def _torch_chamfer_squared(xyz1, xyz2):
    """Return bidirectional squared nearest-neighbour distances in PyTorch.

    The legacy CUDA extension is tied to the PyTorch/CUDA ABI it was built
    with.  Keep a tensor-only implementation available so experiments remain
    reproducible after a framework upgrade.  Processing the first set in
    chunks avoids materialising a full B x N x M distance matrix.
    """
    if xyz1.ndim != 3 or xyz2.ndim != 3 or xyz1.size(0) != xyz2.size(0):
        raise ValueError(
            "Chamfer inputs must be (B, N, 3) and (B, M, 3) with matching B; "
            f"got {tuple(xyz1.shape)} and {tuple(xyz2.shape)}"
        )

    chunk_size = int(os.environ.get("CHAMFER_TORCH_CHUNK_SIZE", "512"))
    if chunk_size <= 0:
        raise ValueError("CHAMFER_TORCH_CHUNK_SIZE must be positive")

    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y.  The clamp only removes
    # round-off-level negative values and preserves the extension's metric.
    xyz2_t = xyz2.transpose(1, 2).contiguous()
    xyz2_norm = xyz2.square().sum(dim=-1).unsqueeze(1)
    dist1_chunks = []
    dist2_chunks = []
    for start in range(0, xyz1.size(1), chunk_size):
        points = xyz1[:, start:start + chunk_size]
        pairwise = (
            points.square().sum(dim=-1, keepdim=True)
            + xyz2_norm
            - 2.0 * torch.bmm(points, xyz2_t)
        ).clamp_min_(0.0)
        dist1_chunks.append(pairwise.amin(dim=2))
        dist2_chunks.append(pairwise.amin(dim=1))

    return torch.cat(dist1_chunks, dim=1), torch.stack(dist2_chunks, dim=1).amin(dim=1)


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


def _chamfer_squared(xyz1, xyz2):
    """Select the maintained tensor backend unless legacy CUDA is requested."""
    backend = os.environ.get("CHAMFER_BACKEND", "torch").lower()
    if backend == "torch":
        return _torch_chamfer_squared(xyz1, xyz2)
    if backend == "cuda":
        if chamfer is None:
            raise RuntimeError(
                "The legacy chamfer extension is unavailable; use "
                "CHAMFER_BACKEND=torch or rebuild it for this PyTorch/CUDA runtime."
            )
        return ChamferFunction.apply(xyz1, xyz2)
    raise ValueError("CHAMFER_BACKEND must be 'torch' or 'cuda'")


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = _chamfer_squared(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)
    
class ChamferDistanceMD(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = _chamfer_squared(xyz1, xyz2)
        return dist1.mean(1) + dist2.mean(1)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = _chamfer_squared(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = _chamfer_squared(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        # The tensor backend can round an almost-zero squared distance down to
        # exactly zero. A raw sqrt then has an infinite derivative, which
        # destabilizes SnowflakeNet's L1 training. The epsilon is far below the
        # reported x1e3 precision but makes the backward pass finite.
        eps = 1e-8
        dist1 = torch.sqrt(dist1 + eps)
        dist2 = torch.sqrt(dist2 + eps)
        return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1_side(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = _chamfer_squared(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        # dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)
    
class ChamferDistanceL2_side(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = _chamfer_squared(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        # dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)
