# SPDX-License-Identifier: Apache-2.0
"""Global Weighted Farthest-Point Sampling (LIMMT Stage III).

Selects a compact, diversity-rich subset from the embedding space, biased toward
dynamically complex motions.  Direct reimplementation of LIMMT Algorithm 1::

    Score(u) = alpha * D_hat(u, S) + (1 - alpha) * C_hat(u)

where ``D_hat`` is the (normalised) distance to the nearest already-selected
clip and ``C_hat`` the rank-normalised complexity.  The anchor is the
highest-complexity clip.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _rank_normalize(values: NDArray) -> NDArray:
    """Map values to [0, 1] by rank (ties share the average rank)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return v
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def global_weighted_fps(
    embeddings: NDArray,
    complexity: NDArray,
    k: int,
    *,
    alpha: float = 0.99,
) -> list[int]:
    """Return indices of the ``k`` selected clips (LIMMT Global Weighted FPS)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    n = emb.shape[0]
    if n == 0 or k <= 0:
        return []
    k = int(min(k, n))

    c_hat = _rank_normalize(np.asarray(complexity, dtype=np.float64))

    # Anchor: hardest (highest complexity) motion.
    anchor = int(np.argmax(c_hat))
    selected = [anchor]
    dist = np.linalg.norm(emb - emb[anchor], axis=1)
    dist[anchor] = -np.inf

    while len(selected) < k:
        d_max = float(np.max(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else 0.0
        d_hat = dist / d_max if d_max > 1e-12 else np.zeros_like(dist)
        score = alpha * d_hat + (1.0 - alpha) * c_hat
        for s in selected:
            score[s] = -np.inf
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        selected.append(nxt)
        new_d = np.linalg.norm(emb - emb[nxt], axis=1)
        dist = np.minimum(dist, new_d)
        for s in selected:
            dist[s] = -np.inf

    return selected


__all__ = ["global_weighted_fps"]
