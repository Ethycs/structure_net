#!/usr/bin/env python3
"""
Patch-Based Network Compactification

Implements the core compactification system with:
- 20% dense patches at extrema locations
- Efficient sparse skeleton representation
- Layer-wise compactification for constant memory
- Hardware-optimized memory layouts

DEPRECATED: This module is deprecated. Please use the new component-based
implementations in structure_net.components instead:
- PatchCompactifier -> components.evolvers.compactification_evolver
- ExtremaDetector -> components.analyzers.compactification_analyzer
- CompactLayer -> components.layers
"""

import warnings

warnings.warn(
    "The compactification.patch_compactification module is deprecated. "
    "Please use structure_net.components.evolvers and analyzers instead.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import scipy.sparse as sp

from ..profiling import profile_component, profile_operation, ProfilerLevel


@dataclass
class PatchInfo:
    """Information about a dense patch."""
    position: Tuple[int, int]  # (row_start, col_start)
    size: Tuple[int, int]      # (height, width)
    density: float
    data: torch.Tensor
    importance_score: float = 0.0


@profile_component(component_name="extrema_detector", level=ProfilerLevel.BASIC)
class ExtremaDetector:
    """
    Detects extrema locations for patch placement.

    Uses gradient analysis and topological features to identify
    critical points where dense patches should be placed.
    """

    def __init__(self, patch_size: int = 8, min_density: float = 0.15):
        self.patch_size = patch_size
        self.min_density = min_density

    def find_extrema_locations(self,
                               weight_matrix: torch.Tensor,
                               target_patches: int) -> List[Tuple[int, int]]:
        """Find optimal locations for dense patches."""
        with profile_operation("extrema_detection", "topology"):
            # Compute gradient magnitude
            grad_magnitude = self._compute_gradient_magnitude(weight_matrix)

            # Find local maxima
            local_maxima = self._find_local_maxima(grad_magnitude)

            # Score potential patch locations
            scored_locations = self._score_patch_locations(
                weight_matrix, local_maxima
            )

            # Select top locations
            selected = sorted(scored_locations,
                              key=lambda x: x[2], reverse=True)
            locations = [(pos[0], pos[1])
                         for pos, _, _ in selected[:target_patches]]

            return locations

    def _compute_gradient_magnitude(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude across the weight matrix."""
        # Sobel operators for gradient computation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=weight_matrix.dtype)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=weight_matrix.dtype)

        # Pad weight matrix
        padded = F.pad(weight_matrix, (1, 1, 1, 1), mode='reflect')

        # Compute gradients
        grad_x = F.conv2d(padded.unsqueeze(0).unsqueeze(0),
                          sobel_x.unsqueeze(0).unsqueeze(0), padding=0)
        grad_y = F.conv2d(padded.unsqueeze(0).unsqueeze(0),
                          sobel_y.unsqueeze(0).unsqueeze(0), padding=0)

        # Magnitude
        magnitude = torch.sqrt(grad_x.squeeze() ** 2 + grad_y.squeeze() ** 2)

        return magnitude

    def _find_local_maxima(self, gradient_magnitude: torch.Tensor) -> List[Tuple[int, int]]:
        """Find local maxima in gradient magnitude."""
        maxima = []
        h, w = gradient_magnitude.shape

        # Use a sliding window to find local maxima
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gradient_magnitude[i, j]

                # Check if it's a local maximum
                neighborhood = gradient_magnitude[i-1:i+2, j-1:j+2]
                if center == neighborhood.max() and center > neighborhood.mean():
                    maxima.append((i, j))

        return maxima

    def _score_patch_locations(self,
                               weight_matrix: torch.Tensor,
                               candidates: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], float, float]]:
        """Score potential patch locations."""
        scored = []
        h, w = weight_matrix.shape

        for row, col in candidates:
            # Check if patch fits
            if (row + self.patch_size <= h and col + self.patch_size <= w):
                # Extract potential patch
                patch = weight_matrix[row:row +
                                      self.patch_size, col:col+self.patch_size]

                # Compute density
                density = (patch.abs() > 1e-6).float().mean().item()

                # Compute importance score
                importance = self._compute_importance_score(patch)

                if density >= self.min_density:
                    scored.append(((row, col), density, importance))

        return scored

    def _compute_importance_score(self, patch: torch.Tensor) -> float:
        """Compute importance score for a patch."""
        # Combine multiple factors
        magnitude_score = patch.abs().mean().item()
        variance_score = patch.var().item()
        connectivity_score = self._compute_connectivity_score(patch)

        # Weighted combination
        importance = (0.4 * magnitude_score +
                      0.3 * variance_score +
                      0.3 * connectivity_score)

        return importance

    def _compute_connectivity_score(self, patch: torch.Tensor) -> float:
        """Compute connectivity score based on non-zero pattern."""
        # Convert to binary
        binary = (patch.abs() > 1e-6).float()

        # Count connected components (simplified)
        total_nonzero = binary.sum().item()
        if total_nonzero == 0:
            return 0.0

        # Estimate connectivity by checking neighbors
        connectivity = 0.0
        h, w = binary.shape

        for i in range(h):
            for j in range(w):
                if binary[i, j] > 0:
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if (di != 0 or dj != 0 and
                                    0 <= i + di < h and 0 <= j + dj < w):
                                if binary[i + di, j + dj] > 0:
                                    neighbors += 1
                    connectivity += neighbors

        return connectivity / (total_nonzero * 8) if total_nonzero > 0 else 0.0


@profile_component(component_name="patch_compactifier", level=ProfilerLevel.BASIC)
class PatchCompactifier:
    """
    Compactifies sparse layers using patch-based representation.

    Stores 20% dense patches separately from sparse skeleton,
    achieving significant memory reduction and cache efficiency.
    """

    def __init__(self, patch_size: int = 8, patch_density: float = 0.2):
        self.patch_size = patch_size
        self.patch_density = patch_density
        self.extrema_detector = ExtremaDetector(
            patch_size, patch_density * 0.75)

    def compactify_layer(self,
                         weight_matrix: torch.Tensor,
                         target_sparsity: float = 0.02) -> Dict[str, Any]:
        """
        Compactify a layer into patches + skeleton representation.

        Args:
            weight_matrix: Full weight matrix to compactify
            target_sparsity: Target overall sparsity

        Returns:
            Compact representation with patches and skeleton
        """
        with profile_operation("layer_compactification", "compression"):
            # Calculate target number of patches
            total_params = weight_matrix.numel()
            target_active = int(total_params * target_sparsity)
            params_per_patch = self.patch_size * self.patch_size * self.patch_density
            target_patches = max(
                1, int(target_active * 0.75 / params_per_patch))

            # Find extrema locations
            extrema_locations = self.extrema_detector.find_extrema_locations(
                weight_matrix, target_patches
            )

            # Extract patches
            patches = self._extract_patches(weight_matrix, extrema_locations)

            # Create skeleton (remaining connections)
            skeleton = self._create_skeleton(
                weight_matrix, patches, target_sparsity)

            # Compute compression statistics
            stats = self._compute_compression_stats(
                weight_matrix, patches, skeleton)

            return {
                'format': 'patch_plus_skeleton',
                'patches': patches,
                'skeleton': skeleton,
                'original_shape': weight_matrix.shape,
                'compression_stats': stats,
                'patch_size': self.patch_size,
                'patch_density': self.patch_density
            }

    def _extract_patches(self,
                         weight_matrix: torch.Tensor,
                         locations: List[Tuple[int, int]]) -> List[PatchInfo]:
        """Extract dense patches at specified locations."""
        patches = []

        for row, col in locations:
            # Extract patch region
            patch_region = weight_matrix[row:row+self.patch_size,
                                         col:col+self.patch_size].clone()

            # Sparsify to target density
            patch_sparse = self._sparsify_patch(
                patch_region, self.patch_density)

            # Create patch info
            patch_info = PatchInfo(
                position=(row, col),
                size=(self.patch_size, self.patch_size),
                density=self.patch_density,
                data=patch_sparse,
                importance_score=self.extrema_detector._compute_importance_score(
                    patch_sparse)
            )

            patches.append(patch_info)

            # Zero out this region in original matrix
            weight_matrix[row:row+self.patch_size, col:col+self.patch_size] = 0

        return patches

    def _sparsify_patch(self, patch: torch.Tensor, target_density: float) -> torch.Tensor:
        """Sparsify patch to target density while preserving important connections."""
        # Flatten and get magnitudes
        flat_patch = patch.flatten()
        magnitudes = flat_patch.abs()

        # Keep top-k elements
        k = int(len(flat_patch) * target_density)
        if k == 0:
            return torch.zeros_like(patch)

        # Find threshold
        threshold = torch.topk(magnitudes, k).values[-1]

        # Create sparse patch
        mask = magnitudes >= threshold
        sparse_flat = torch.zeros_like(flat_patch)
        sparse_flat[mask] = flat_patch[mask]

        return sparse_flat.reshape(patch.shape)

    def _create_skeleton(self,
                         weight_matrix: torch.Tensor,
                         patches: List[PatchInfo],
                         target_sparsity: float) -> sp.csr_matrix:
        """Create sparse skeleton from remaining connections."""
        # Calculate remaining budget
        total_params = weight_matrix.numel()
        patch_params = sum(p.data.count_nonzero().item() for p in patches)
        remaining_budget = int(total_params * target_sparsity) - patch_params

        if remaining_budget <= 0:
            # No budget for skeleton
            return sp.csr_matrix(weight_matrix.shape)

        # Find top remaining connections
        flat_weights = weight_matrix.flatten()
        magnitudes = flat_weights.abs()

        # Get top-k remaining connections
        k = min(remaining_budget, (magnitudes > 0).sum().item())
        if k > 0:
            threshold = torch.topk(magnitudes, k).values[-1]
            mask = magnitudes >= threshold

            # Create sparse matrix
            sparse_flat = torch.zeros_like(flat_weights)
            sparse_flat[mask] = flat_weights[mask]
            sparse_matrix = sparse_flat.reshape(weight_matrix.shape)

            # Convert to scipy sparse format
            skeleton = sp.csr_matrix(sparse_matrix.cpu().numpy())
        else:
            skeleton = sp.csr_matrix(weight_matrix.shape)

        return skeleton

    def _compute_compression_stats(self,
                                   original: torch.Tensor,
                                   patches: List[PatchInfo],
                                   skeleton: sp.csr_matrix) -> Dict[str, Any]:
        """Compute compression statistics."""
        original_size = original.numel() * 4  # float32

        # Patch storage
        patch_data_size = sum(p.data.numel() * 4 for p in patches)
        patch_metadata_size = len(patches) * 32  # positions, etc.

        # Skeleton storage
        skeleton_size = (skeleton.data.nbytes +
                         skeleton.indices.nbytes +
                         skeleton.indptr.nbytes)

        total_compressed = patch_data_size + patch_metadata_size + skeleton_size

        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': total_compressed,
            'compression_ratio': original_size / total_compressed if total_compressed > 0 else float('inf'),
            'patch_count': len(patches),
            'patch_storage_bytes': patch_data_size + patch_metadata_size,
            'skeleton_storage_bytes': skeleton_size,
            'skeleton_nnz': skeleton.nnz
        }

    def reconstruct_layer(self, compact_data: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct full layer from compact representation."""
        shape = compact_data['original_shape']
        reconstructed = torch.zeros(shape)

        # Place patches
        for patch in compact_data['patches']:
            row, col = patch.position
            h, w = patch.size
            reconstructed[row:row+h, col:col+w] = patch.data

        # Add skeleton
        skeleton_tensor = torch.from_numpy(
            compact_data['skeleton'].toarray()).float()
        reconstructed += skeleton_tensor

        return reconstructed


@profile_component(component_name="compact_layer", level=ProfilerLevel.BASIC)
class CompactLayer(nn.Module):
    """
    Compact layer implementation with patches and skeleton.

    Efficiently computes forward pass using dense patches
    and sparse skeleton without reconstructing full matrix.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 sparsity: float = 0.02,
                 patch_density: float = 0.2,
                 patch_size: int = 8,
                 patch_locations: Optional[List[int]] = None,
                 avoid_connections: Optional[torch.Tensor] = None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.patch_density = patch_density
        self.patch_size = patch_size

        # Create initial weight matrix
        initial_weight = self._create_initial_weight(
            patch_locations, avoid_connections)

        # Compactify immediately
        self.compactifier = PatchCompactifier(patch_size, patch_density)
        self.compact_data = self.compactifier.compactify_layer(
            initial_weight, sparsity)

        # Store patches as parameters
        self.patches = nn.ParameterList()
        self.patch_positions = []

        for patch_info in self.compact_data['patches']:
            self.patches.append(nn.Parameter(patch_info.data))
            self.patch_positions.append(patch_info.position)

        # Store skeleton (non-trainable for now, can be made trainable)
        self.register_buffer('skeleton_data',
                             torch.from_numpy(self.compact_data['skeleton'].data).float())
        self.register_buffer('skeleton_indices',
                             torch.from_numpy(self.compact_data['skeleton'].indices).long())
        self.register_buffer('skeleton_indptr',
                             torch.from_numpy(self.compact_data['skeleton'].indptr).long())

        # Bias
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def _create_initial_weight(self,
                               patch_locations: Optional[List[int]],
                               avoid_connections: Optional[torch.Tensor]) -> torch.Tensor:
        """Create initial weight matrix with guided structure."""
        weight = torch.zeros(self.output_dim, self.input_dim)

        # Initialize with small random values
        nn.init.normal_(weight, std=0.01)

        # Zero out connections to avoid
        if avoid_connections is not None and len(avoid_connections) > 0:
            weight[:, avoid_connections] = 0

        # Enhance regions where patches will be placed
        if patch_locations is not None:
            for loc in patch_locations:
                if loc < self.input_dim - self.patch_size:
                    # Slightly boost this region
                    weight[:, loc:loc+self.patch_size] *= 1.5

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient forward pass using compact representation."""
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Apply patches (dense operations)
        for i, (patch, position) in enumerate(zip(self.patches, self.patch_positions)):
            row_start, col_start = position
            row_end = row_start + self.patch_size
            col_end = col_start + self.patch_size

            # Extract input region
            if col_end <= x.shape[1]:
                x_patch = x[:, col_start:col_end]

                # Dense matrix multiplication
                patch_output = F.linear(x_patch, patch)

                # Add to output
                if row_end <= output.shape[1]:
                    output[:, row_start:row_end] += patch_output

        # Apply skeleton (sparse operation)
        if len(self.skeleton_data) > 0:
            skeleton_output = self._sparse_matmul(x, self.skeleton_data,
                                                  self.skeleton_indices,
                                                  self.skeleton_indptr)
            output += skeleton_output

        # Add bias
        output += self.bias

        return output

    def _sparse_matmul(self, x: torch.Tensor, data: torch.Tensor,
                       indices: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
        """Efficient sparse matrix multiplication."""
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Simple implementation (can be optimized with custom CUDA kernel)
        for row in range(self.output_dim):
            start_idx = indptr[row]
            end_idx = indptr[row + 1]

            if start_idx < end_idx:
                cols = indices[start_idx:end_idx]
                values = data[start_idx:end_idx]

                # Compute dot product
                output[:, row] = torch.sum(
                    x[:, cols] * values.unsqueeze(0), dim=1)

        return output

    def reconstruct_full_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix (for analysis purposes)."""
        weight = torch.zeros(self.output_dim, self.input_dim)

        # Place patches
        for patch, position in zip(self.patches, self.patch_positions):
            row_start, col_start = position
            row_end = row_start + self.patch_size
            col_end = col_start + self.patch_size

            weight[row_start:row_end, col_start:col_end] = patch.data

        # Add skeleton
        for row in range(self.output_dim):
            start_idx = self.skeleton_indptr[row]
            end_idx = self.skeleton_indptr[row + 1]

            if start_idx < end_idx:
                cols = self.skeleton_indices[start_idx:end_idx]
                values = self.skeleton_data[start_idx:end_idx]
                weight[row, cols] = values

        return weight

    def get_compression_info(self) -> Dict[str, Any]:
        """Get compression information for this layer."""
        return {
            'compression_stats': self.compact_data['compression_stats'],
            'patch_count': len(self.patches),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'patch_parameters': sum(p.numel() for p in self.patches),
            'skeleton_parameters': len(self.skeleton_data)
        }


def create_compact_layer(input_dim: int,
                         output_dim: int,
                         sparsity: float = 0.02,
                         **kwargs) -> CompactLayer:
    """
    Factory function to create a compact layer.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        sparsity: Target sparsity level
        **kwargs: Additional arguments for CompactLayer

    Returns:
        Configured CompactLayer
    """
    return CompactLayer(
        input_dim=input_dim,
        output_dim=output_dim,
        sparsity=sparsity,
        **kwargs
    )
