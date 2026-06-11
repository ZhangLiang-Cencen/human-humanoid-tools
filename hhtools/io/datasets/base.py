"""Dataset adapter protocol and registry.

Every open-source motion dataset we support (AMASS, GVHMR, Motion-X, PHUMA, LAFAN, SOMA, ...)
is exposed via a :class:`DatasetAdapter` subclass.  Adapters differ in the *raw* on-disk format
but share a common surface:

1. :meth:`DatasetAdapter.list_sequences` enumerates the sequence ids available under ``root``.
2. :meth:`DatasetAdapter.load_params` returns a :class:`SmplMotionParams` for SMPL-based
   datasets (raises :class:`NotImplementedError` for BVH / FBX based datasets).
3. :meth:`DatasetAdapter.load_motion` returns a :class:`Motion` by either running the SMPL
   forward engine on the params, or directly parsing the skeletal animation.

The adapter layer never forces the engine to load -- a consumer that only needs metadata can
call :meth:`list_sequences` without any MPI weights being present on disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from hhtools.core.motion import Motion

# String tag identifying which body model family / mode the adapter needs.
DatasetMode = Literal["smpl", "smplh", "smplx", "bvh", "skeleton"]


class DatasetAdapter(ABC):
    """Interface for reading a public human motion dataset.

    Attributes:
        name: Short machine-readable dataset name (e.g. ``"amass"`` or ``"motion_x"``).
        display_name: Pretty name shown in UIs.
        requires: Which optional body model family the adapter needs. One of
            ``"smpl"``, ``"smplh"``, ``"smplx"``, ``"bvh"``, ``"skeleton"`` (raw joint data).
        file_patterns: Glob patterns used to discover sequence files under ``root``.
    """

    name: str = ""
    display_name: str = ""
    requires: DatasetMode = "skeleton"
    file_patterns: tuple[str, ...] = ()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @abstractmethod
    def list_sequences(self) -> Iterator[str]:
        """Yield sequence identifiers (relative paths or stable ids) under ``self.root``."""

    def load_params(self, sequence_id: str):  # noqa: ANN201 -- SmplMotionParams (lazy import)
        """Extract SMPL / SMPL-H / SMPL-X parameters for *sequence_id*.

        Default implementation raises :class:`NotImplementedError`; BVH / skeleton adapters
        override :meth:`load_motion` directly instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose SMPL parameters. "
            f"Call load_motion() directly."
        )

    @abstractmethod
    def load_motion(self, sequence_id: str, **kwargs) -> Motion:
        """Load a single sequence and return it as a :class:`Motion`."""


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_dataset(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Decorator / function to register a dataset adapter."""
    if not cls.name:
        raise ValueError(f"DatasetAdapter subclass {cls.__name__} must set `name`")
    _REGISTRY[cls.name] = cls
    return cls


def registered_datasets() -> dict[str, type[DatasetAdapter]]:
    return dict(_REGISTRY)


def get_dataset(name: str) -> type[DatasetAdapter]:
    """Look up a registered adapter class by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown dataset {name!r}. Registered: {known}") from exc


__all__ = [
    "DatasetAdapter",
    "DatasetMode",
    "get_dataset",
    "register_dataset",
    "registered_datasets",
]
