"""Bone hierarchy descriptor.

A ``Hierarchy`` stores the immutable topology of a character skeleton: bone names, parent
relationships, and the name-to-index lookup used by IO and analytics. It is intentionally free of
per-frame pose data (see :class:`hhtools.core.motion.Motion`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Hierarchy:
    """Immutable bone topology.

    Attributes:
        bone_names: Ordered list of bone names.
        parent_indices: ``int32`` array of length ``num_bones``; ``-1`` marks the root.
        parent_names: Parallel list of parent bone names; the root's entry is ``None``.
    """

    bone_names: list[str]
    parent_indices: NDArray
    parent_names: list[str | None]

    def __post_init__(self) -> None:
        if len(self.bone_names) != len(self.parent_names):
            raise ValueError(
                f"bone_names ({len(self.bone_names)}) and parent_names "
                f"({len(self.parent_names)}) must match"
            )
        self.parent_indices = np.asarray(self.parent_indices, dtype=np.int32)
        if self.parent_indices.shape != (len(self.bone_names),):
            raise ValueError(
                f"parent_indices shape {self.parent_indices.shape} must equal "
                f"({len(self.bone_names)},)"
            )
        self._name_to_index: dict[str, int] = {n: i for i, n in enumerate(self.bone_names)}

    # ---------------------------------------------------------------- constructors

    @classmethod
    def from_parent_names(
        cls, bone_names: Sequence[str], parent_names: Sequence[str | None]
    ) -> Hierarchy:
        """Build a Hierarchy from ``(bone_names, parent_names)``.

        Useful for BVH/GLB importers that know names but not indices.
        """
        name_to_index = {n: i for i, n in enumerate(bone_names)}
        parent_indices = np.full(len(bone_names), -1, dtype=np.int32)
        for i, pname in enumerate(parent_names):
            if pname is None:
                continue
            if pname not in name_to_index:
                raise KeyError(f"Parent bone {pname!r} for {bone_names[i]!r} not found")
            parent_indices[i] = name_to_index[pname]
        return cls(list(bone_names), parent_indices, list(parent_names))

    @classmethod
    def from_parent_indices(
        cls, bone_names: Sequence[str], parent_indices: Iterable[int]
    ) -> Hierarchy:
        """Build a Hierarchy from ``(bone_names, parent_indices)``."""
        pi = np.asarray(list(parent_indices), dtype=np.int32)
        parent_names: list[str | None] = []
        for idx in pi.tolist():
            if idx < 0:
                parent_names.append(None)
            else:
                parent_names.append(bone_names[idx])
        return cls(list(bone_names), pi, parent_names)

    # ----------------------------------------------------------------- queries

    @property
    def num_bones(self) -> int:
        return len(self.bone_names)

    def index(self, name: str) -> int:
        """Return the index of ``name`` or ``-1`` if not present."""
        return self._name_to_index.get(name, -1)

    def indices(self, names: Iterable[str]) -> list[int]:
        return [self.index(n) for n in names]

    def name(self, index: int) -> str:
        if not 0 <= index < self.num_bones:
            raise IndexError(f"Bone index {index} out of range (num_bones={self.num_bones})")
        return self.bone_names[index]

    def parent(self, index: int) -> int:
        return int(self.parent_indices[index])

    def children(self, index: int) -> list[int]:
        return [i for i, p in enumerate(self.parent_indices) if int(p) == index]

    def root_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.parent_indices) if int(p) == -1]

    def is_root(self, index: int) -> bool:
        return self.parent(index) == -1

    def descendants(self, index: int) -> list[int]:
        """Return all descendants of ``index`` in depth-first order (excluding itself)."""
        out: list[int] = []
        stack = list(self.children(index))
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(self.children(i))
        return out

    # ----------------------------------------------------------------- debug

    def debug(self) -> str:
        """Return a human-readable tree representation."""
        lines: list[str] = [f"Hierarchy: {self.num_bones} bones"]

        def walk(idx: int, depth: int) -> None:
            lines.append("  " * depth + f"- [{idx}] {self.bone_names[idx]}")
            for c in self.children(idx):
                walk(c, depth + 1)

        for root in self.root_indices():
            walk(root, 0)
        return "\n".join(lines)
