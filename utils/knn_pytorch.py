"""
PyTorch-based KNN implementation to replace knn_cuda.KNN.
Provides same interface: KNN(k=..., transpose_mode=True)

PointMamba uses transpose_mode=True and passes tensors in (B, N, 3) format.
"""
import torch
import torch.nn as nn


class KNN(nn.Module):
    """KNN (k-nearest neighbors) implemented in pure PyTorch.

    Args:
        k: Number of nearest neighbors
        transpose_mode: If True, expects (B, N, 3) (PointMamba's usage).
                       If False, expects (B, 3, N).
    """
    def __init__(self, k: int, transpose_mode: bool = True):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, xyz: torch.Tensor, query: torch.Tensor):
        """Find k nearest neighbors from xyz for each point in query.

        Args:
            xyz: Source points. (B, N, 3) if transpose_mode=True
            query: Query points. (B, M, 3) if transpose_mode=True

        Returns:
            dists: (B, M, k) distances to nearest neighbors
            idx: (B, M, k) indices of nearest neighbors
        """
        if not self.transpose_mode:
            # transpose_mode=False: input is (B, 3, N) -> convert to (B, N, 3)
            xyz = xyz.transpose(1, 2).contiguous()
            query = query.transpose(1, 2).contiguous()

        # Now both are (B, N, 3) and (B, M, 3)
        # Compute pairwise squared distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b

        # Squared norms: (B, N) and (B, M)
        xyz_sq = (xyz ** 2).sum(dim=-1)      # (B, N)
        query_sq = (query ** 2).sum(dim=-1)   # (B, M)

        # Cross term via batch matmul: (B, M, N)
        # query: (B, M, 3), xyz^T: (B, 3, N) -> (B, M, N)
        cross_term = torch.bmm(query, xyz.transpose(1, 2))

        # Distances: (B, M, N)
        dists = query_sq.unsqueeze(-1) + xyz_sq.unsqueeze(1) - 2 * cross_term
        dists = torch.clamp(dists, min=0.0)

        # Top-k smallest: (B, M, k)
        dists, idx = torch.topk(dists, k=self.k, dim=-1, largest=False)

        return dists, idx
