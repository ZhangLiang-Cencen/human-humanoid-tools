# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
#
# Laplacian mesh helpers adapted from holosoma_retargeting (Apache-2.0).
# See project NOTICE for attribution.
"""Interaction-mesh geometry: Delaunay tetrahedra, adjacency, Laplacian matrix.

Pure NumPy / SciPy. Used by the Laplacian MPC+SQP retargeting backend.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay


def create_interaction_mesh(vertices: NDArray[np.floating]) -> tuple[NDArray, NDArray[np.integer]]:
    """Delaunay tetrahedralization of ``vertices`` (N, 3).

    Returns:
        vertices unchanged (N, 3) and ``simplices`` (T, 4) integer tetra indices.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3); got {v.shape}")
    if v.shape[0] < 4:
        raise ValueError("Delaunay in 3D requires at least 4 vertices")
    tri = Delaunay(v)
    return v.astype(np.float32, copy=False), np.asarray(tri.simplices, dtype=np.int64)


def get_adjacency_list(tetrahedra: NDArray[np.integer], num_vertices: int) -> list[list[int]]:
    """Undirected adjacency from tetra mesh (each edge of each simplex)."""
    adj: list[set[int]] = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = int(tet[i]), int(tet[j])
                adj[u].add(v)
                adj[v].add(u)
    return [sorted(s) for s in adj]


def calculate_laplacian_coordinates(
    vertices: NDArray[np.floating],
    adj_list: list[list[int]],
    *,
    epsilon: float = 1e-6,
    uniform_weight: bool = True,
) -> NDArray[np.float32]:
    """Per-vertex Laplacian δ-coordinates (N, 3), cotangent-free umbrella operator."""
    v = np.asarray(vertices, dtype=np.float64)
    lap = np.zeros_like(v)
    for i in range(len(v)):
        nbr = adj_list[i]
        if not nbr:
            continue
        vi = v[i]
        npos = v[nbr]
        if uniform_weight:
            center = np.mean(npos, axis=0)
        else:
            dist = np.linalg.norm(vi - npos, axis=1)
            w = 1.0 / (1.5 * dist + epsilon)
            center = np.sum(w[:, None] * npos, axis=0) / np.sum(w)
        lap[i] = vi - center
    return lap.astype(np.float32, copy=False)


def calculate_laplacian_matrix(
    vertices: NDArray[np.floating],
    adj_list: list[list[int]],
    *,
    epsilon: float = 1e-6,
    uniform_weight: bool = True,
) -> NDArray[np.float64]:
    """Dense (N, N) Laplacian operator matching holosoma uniform / distance weights."""
    v = np.asarray(vertices, dtype=np.float64)
    n = len(v)
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        nbr = adj_list[i]
        if not nbr:
            continue
        if uniform_weight:
            w = np.ones(len(nbr)) / len(nbr)
        else:
            dist = np.linalg.norm(v[i] - v[nbr], axis=1)
            w = 1.0 / (dist + epsilon)
            w = w / np.sum(w)
        L[i, i] = 1.0
        for j, idx in enumerate(nbr):
            L[i, idx] = -w[j]
    return L


__all__ = [
    "calculate_laplacian_coordinates",
    "calculate_laplacian_matrix",
    "create_interaction_mesh",
    "get_adjacency_list",
]
