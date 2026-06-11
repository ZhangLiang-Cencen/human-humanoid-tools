# SPDX-License-Identifier: Apache-2.0
"""Clip-level embeddings for diversity / similarity (LIMMT Stage II analogue).

The embedding is **not** a quality gate.  It produces a metric space where
distances reflect behavioural similarity, enabling the 2-D scatter
(:mod:`hhtools.analysis.cluster`) and diversity-aware subset selection
(:mod:`hhtools.analysis.subset`).

Two interchangeable backends:

* :class:`HandcraftedEmbedding` (**A**, default) — a fixed vector of the most
  discriminative L0 metrics, z-scored across the collection.  Zero training,
  pure numpy, fully interpretable.
* :class:`PaeEmbedding` (**B**, reserved) — a LIMMT-style Periodic Autoencoder
  trained on the *current* collection's canonical windows, taking the
  phase-invariant ``mean([A, F])`` global descriptor.  Lazily imports ``torch``;
  only built when explicitly requested.

Select via :func:`make_embedding`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Metrics that make up the handcrafted descriptor (behaviour-discriminative,
# source-agnostic).  Missing keys default to 0.
_HANDCRAFTED_KEYS: tuple[str, ...] = (
    "complexity",
    "joint_kinetic_energy",
    "joint_accel_energy",
    "root_speed_xy",
    "root_speed_xy_p95",
    "root_speed_z",
    "root_turn_rate",
    "com_height_std",
    "com_height_range",
    "airborne_ratio",
    "path_efficiency",
    "step_freq",
    "leg_energy",
    "arm_energy",
    "inverted_ratio",
)


class EmbeddingBackend(ABC):
    """Fit on a collection, then encode each clip to a fixed-length vector."""

    name: str = "base"

    @abstractmethod
    def fit(self, clips: list[Any]) -> None:
        ...

    @abstractmethod
    def encode(self, clip: Any) -> NDArray:
        ...

    def fit_encode(self, clips: list[Any]) -> list[NDArray]:
        self.fit(clips)
        return [self.encode(c) for c in clips]


class HandcraftedEmbedding(EmbeddingBackend):
    """Z-scored handcrafted metric vector (archetype A, no training)."""

    name = "handcrafted"

    def __init__(self, keys: tuple[str, ...] = _HANDCRAFTED_KEYS) -> None:
        self.keys = keys
        self._mean: NDArray | None = None
        self._std: NDArray | None = None

    def _raw(self, clip: Any) -> NDArray:
        m = clip.metrics or {}
        return np.array([float(m.get(k, 0.0) or 0.0) for k in self.keys], dtype=np.float64)

    def fit(self, clips: list[Any]) -> None:
        if not clips:
            self._mean = np.zeros(len(self.keys))
            self._std = np.ones(len(self.keys))
            return
        mat = np.stack([self._raw(c) for c in clips], axis=0)
        # log1p-compress heavy-tailed energy terms before z-scoring.
        mat = np.sign(mat) * np.log1p(np.abs(mat))
        self._mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        std[std < 1e-6] = 1.0
        self._std = std

    def encode(self, clip: Any) -> NDArray:
        if self._mean is None or self._std is None:
            raise RuntimeError("HandcraftedEmbedding.fit must be called before encode")
        raw = self._raw(clip)
        raw = np.sign(raw) * np.log1p(np.abs(raw))
        z = (raw - self._mean) / self._std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class PaeEmbedding(EmbeddingBackend):
    """LIMMT-style Periodic Autoencoder embedding (archetype B, reserved).

    Trains a small PAE on 4 s canonical windows of the current collection using a
    reconstruction loss only, then takes the phase-invariant ``mean([A, F])``
    global descriptor per clip.  Implementation is intentionally deferred; the
    class exists so :func:`make_embedding` and the UI can switch backends without
    touching callers.  Requires the optional ``torch`` dependency.
    """

    name = "pae"

    def __init__(self, window_s: float = 4.0, k: int = 8) -> None:
        self.window_s = window_s
        self.k = k

    def fit(self, clips: list[Any]) -> None:  # pragma: no cover - reserved path
        raise NotImplementedError(
            "PAE embedding (backend B) is reserved. Use backend 'handcrafted' for "
            "now; the PAE trainer will train on the current collection's canonical "
            "windows with a reconstruction loss and emit mean([A, F]) descriptors."
        )

    def encode(self, clip: Any) -> NDArray:  # pragma: no cover - reserved path
        raise NotImplementedError(self.fit.__doc__)


def make_embedding(name: str, cfg: dict[str, Any] | None = None) -> EmbeddingBackend:
    """Factory: ``"handcrafted"`` (A) or ``"pae"`` (B)."""
    name = (name or "handcrafted").lower()
    if name in ("handcrafted", "a", "manual"):
        return HandcraftedEmbedding()
    if name in ("pae", "b", "hme"):
        pae_cfg = (cfg or {}).get("embedding", {}).get("pae", {})
        return PaeEmbedding(
            window_s=float(pae_cfg.get("window_s", 4.0)),
            k=int(pae_cfg.get("k", 8)),
        )
    raise ValueError(f"unknown embedding backend {name!r}")


__all__ = ["EmbeddingBackend", "HandcraftedEmbedding", "PaeEmbedding", "make_embedding"]
