"""Adapter for directories of pre-converted unified NPZ files.

These directories contain ``.npz`` files already in the hhtools unified schema
(e.g. exported from PARC MS ``.pkl`` via ``scripts/dec_release_to_parc_ms_clips.py``).
No conversion or body-model engine is needed — ``load_npz`` handles everything.

Sibling mesh files (terrain ``.obj``, etc.) referenced via ``objects_mesh_paths``
are resolved relative to the NPZ's parent directory at load time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hhtools.core.motion import Motion
from hhtools.io.datasets.base import DatasetAdapter, register_dataset


@register_dataset
class UnifiedNpzFolderAdapter(DatasetAdapter):
    """Browse a directory of pre-converted unified ``.npz`` clips."""

    name = "unified_npz"
    display_name = "Unified NPZ"
    requires = "skeleton"
    file_patterns = ("*.npz",)

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.npz")):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        path = (self.root / sequence_id).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"unified_npz clip not found: {path}")
        from hhtools.io.loader_progress import pop_progress_callback
        from hhtools.io.npz import load_npz

        progress_callback = pop_progress_callback(kwargs)
        m = load_npz(path, progress_callback=progress_callback)
        m.meta["dataset"] = self.name
        m.meta["sequence_id"] = sequence_id
        return m


__all__ = ["UnifiedNpzFolderAdapter"]
