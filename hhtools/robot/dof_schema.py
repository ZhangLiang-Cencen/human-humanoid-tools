"""CSV schema generator for robot trajectories (xyz + xyzw + dof).

The hhtools-canonical robot CSV format is:

    time, root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw,
    dof_<joint1>, dof_<joint2>, ..., dof_<jointN>

This module owns that format.  Two main entry points:

* :func:`header_columns` — returns the column names for a given robot model.
* :func:`write_empty_csv` — writes just the header, useful for "export-schema"
  UX flows where the user wants to see what a valid CSV looks like before
  populating it.

We deliberately *don't* do any numeric I/O here — filling rows with real
retargeted data is ``hhtools/io/csv.py``'s job.  This module is pure metadata
so the UI can query "what DOFs does this robot expose?" without importing the
retargeter or pandas.
"""

from __future__ import annotations

from pathlib import Path

from hhtools.robot.base import RobotModel

# Floating-base header layout.  Seven columns after ``time``: three for
# position, four for quaternion in xyzw order (matches the rest of hhtools
# and soma-retargeter).  If we ever add fixed-base robots we'll toggle the
# floating block via ``preset.meta["floating_base"]`` — for now every robot
# we support is a humanoid with a free-floating pelvis, so we hard-code it.
_ROOT_COLUMNS: tuple[str, ...] = (
    "root_x", "root_y", "root_z",
    "root_qx", "root_qy", "root_qz", "root_qw",
)


def header_columns(robot: RobotModel) -> list[str]:
    """Return the ordered list of CSV columns for ``robot``.

    Structure: ``["time", <7 root columns>, "dof_<joint1>", ..., "dof_<jointN>"]``.
    The DOF order comes from ``robot.actuated_joints`` which in turn honours
    ``robot.yaml.dof_order`` — see :func:`hhtools.robot.loader._order_actuated`.
    """
    dof_columns = [f"dof_{name}" for name in robot.dof_names()]
    return ["time", *_ROOT_COLUMNS, *dof_columns]


def header_csv(robot: RobotModel) -> str:
    """Return the header row as a CSV-encoded string (no trailing newline).

    Use this when you want to embed the schema inline somewhere (e.g. a log
    message) without opening a file.
    """
    return ",".join(header_columns(robot))


def write_empty_csv(robot: RobotModel, path: str | Path) -> Path:
    """Write just the header row of ``robot``'s CSV schema to ``path``.

    Useful for:
    * Showing users what columns their retargeted output will carry.
    * Smoke-testing the loader — if writing the schema succeeds, the URDF
      parsed and the dof_order validated.

    Returns the resolved path.  Existing files are overwritten.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header_csv(robot) + "\n", encoding="utf-8")
    return target
