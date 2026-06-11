# SPDX-License-Identifier: Apache-2.0
"""2-D projection + clustering of clip embeddings for the scatter view.

* **Projection** to 2-D for the scatter plot: UMAP if installed, else PCA
  (``scikit-learn`` is a hard hhtools dependency, so PCA always works).
* **Clustering** into discrete groups: HDBSCAN if installed, else KMeans with a
  heuristic ``k``.

Both degrade gracefully so the feature works out of the box without optional
extras.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def project_2d(embeddings: NDArray) -> NDArray:
    """Project ``(N, D)`` embeddings to ``(N, 2)`` scatter coordinates."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return np.zeros((emb.shape[0] if emb.ndim else 0, 2), dtype=np.float32)
    if emb.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if emb.shape[1] <= 2:
        out = np.zeros((emb.shape[0], 2), dtype=np.float32)
        out[:, : emb.shape[1]] = emb[:, :2]
        return out

    try:  # optional: nicer manifold layout
        import umap  # type: ignore

        n_neighbors = min(15, max(2, emb.shape[0] - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        return reducer.fit_transform(emb).astype(np.float32)
    except Exception:
        pass

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(emb).astype(np.float32)


def cluster(embeddings: NDArray, *, max_k: int = 12) -> NDArray:
    """Cluster ``(N, D)`` embeddings -> integer label per row (``-1`` = noise)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    n = emb.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n < 3:
        return np.zeros((n,), dtype=np.int32)

    try:  # optional: density-based, finds noise + variable cluster count
        import hdbscan  # type: ignore

        min_cluster = max(3, n // 50)
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster).fit_predict(emb)
        return labels.astype(np.int32)
    except Exception:
        pass

    from sklearn.cluster import KMeans

    k = int(min(max_k, max(2, round(np.sqrt(n / 2)))))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)
    return labels.astype(np.int32)


__all__ = ["cluster", "project_2d"]
