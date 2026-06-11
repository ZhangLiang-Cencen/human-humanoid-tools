"""Robot side of hhtools: URDF loading, preset registry, DOF schema.

Public surface:

* :class:`RobotModel` / :class:`URDFRobotModel` — the thing IK and rendering
  talk to.
* :class:`RobotPreset` — deserialised ``robot.yaml``.
* :func:`load_robot` — URDF + MJCF materialisation.
* :func:`list_presets` / :func:`get` — registry access.
* :func:`header_columns` / :func:`write_empty_csv` — CSV schema helpers.

The submodules are deliberately import-light.  Importing ``hhtools.robot``
alone does not touch yourdfpy/mujoco; those come in only when you actually
call :func:`load_robot`.
"""

from __future__ import annotations

import importlib

from hhtools.robot.base import JointInfo, LinkInfo, RobotModel, RobotPreset
from hhtools.robot.dof_schema import header_columns, header_csv, write_empty_csv
from hhtools.robot.registry import clear_cache, get, list_presets, refresh
from hhtools.robot.scaffold import ScaffoldResult, scaffold_preset, scaffold_yaml_file

__all__ = [
    "JointInfo",
    "LinkInfo",
    "RobotModel",
    "RobotPreset",
    "ScaffoldResult",
    "URDFRobotModel",
    "clear_cache",
    "get",
    "header_columns",
    "header_csv",
    "list_presets",
    "load_robot",
    "refresh",
    "scaffold_preset",
    "scaffold_yaml_file",
    "write_empty_csv",
]


def __getattr__(name: str):
    """Load MuJoCo-backed symbols only on demand (matches package docstring)."""
    if name in ("URDFRobotModel", "load_robot"):
        mod = importlib.import_module("hhtools.robot.loader")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
