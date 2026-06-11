"""Dataset adapter registry and built-in adapters.

Importing this module auto-registers adapters for every dataset hhtools ships out of the box.
Individual adapters keep their heavy dependencies (``smplx``, ``torch``) lazy so that simply
listing sequences does not force the SMPL engine to load.
"""

from __future__ import annotations

from hhtools.io.datasets.base import (
    DatasetAdapter,
    DatasetMode,
    get_dataset,
    register_dataset,
    registered_datasets,
)

# Side-effectful imports: each module registers its adapter class with the registry.
from hhtools.io.datasets import amass as _amass  # noqa: F401
from hhtools.io.datasets import bvh_folder as _bvh_folder  # noqa: F401
from hhtools.io.datasets import glb_folder as _glb_folder  # noqa: F401
from hhtools.io.datasets import hmr4d as _hmr4d  # noqa: F401
from hhtools.io.datasets import meshmimic_holosoma as _meshmimic_holosoma  # noqa: F401
from hhtools.io.datasets import motion_x as _motion_x  # noqa: F401
from hhtools.io.datasets import omomo as _omomo  # noqa: F401
from hhtools.io.datasets import parc_ms as _parc_ms  # noqa: F401
from hhtools.io.datasets import phuma as _phuma  # noqa: F401
from hhtools.io.datasets import unified_npz_folder as _unified_npz_folder  # noqa: F401

__all__ = [
    "DatasetAdapter",
    "DatasetMode",
    "get_dataset",
    "register_dataset",
    "registered_datasets",
]
