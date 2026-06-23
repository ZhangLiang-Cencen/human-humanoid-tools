"""CSV reader / writer for retargeted robot trajectories.

The canonical column layout is defined in
:mod:`hhtools.robot.dof_schema`::

    time, root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw,
    dof_<joint1>, dof_<joint2>, ...

This module owns the numeric IO.  It is intentionally thin — no pandas
dependency — because the CSVs are small (a few minutes at 30 Hz ≈ a few
kilobytes).  The writer fills rows from a ``(F, 7 + N)`` array; the reader
returns the same shape + header metadata for the UI to sanity-check against
the target robot preset.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hhtools.robot.base import RobotModel
from hhtools.robot.dof_schema import header_columns


__all__ = ["RobotCSV", "load_robot_csv", "save_robot_csv"]


@dataclass
class RobotCSV:
    """Parsed robot trajectory CSV.

    ``joint_q`` has shape ``(F, 7 + N)`` where the first 7 columns are the
    floating-base root and the next ``N`` are actuated DOFs, in the order
    the header declared.  ``times`` is a separate ``(F,)`` array.
    """

    times: NDArray
    joint_q: NDArray
    dof_names: tuple[str, ...]
    sample_rate: float
    meta: dict

    @property
    def num_frames(self) -> int:
        return int(self.joint_q.shape[0])


def save_robot_csv(
    path: str | Path,
    *,
    robot: RobotModel,
    joint_q: NDArray,
    sample_rate: float,
    times: NDArray | None = None,
    meta: dict | None = None,
    include_header: bool = True,
) -> Path:
    """Write a retargeted trajectory to a CSV at ``path``.

    Args:
        path: Target CSV file; parent directories are created on demand.
        robot: The :class:`RobotModel` — used for the header / DOF order.
        joint_q: ``(F, 7 + N)`` array aligned with ``header_columns(robot)[1:]``.
        sample_rate: Frames per second; recorded as a header comment so
            readers (and the user) can reconstruct the timebase.
        times: Explicit ``(F,)`` array of timestamps in seconds.  ``None``
            derives ``t = frame / sample_rate``.
        meta: Optional extra key/value pairs rendered as ``# key: value``
            comment lines at the top of the file.  Useful for embedding
            robot preset name, source motion, IK config, etc.

    Returns the resolved path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    header = header_columns(robot)
    expected_cols = len(header) - 1  # minus "time"
    if joint_q.ndim != 2 or joint_q.shape[1] != expected_cols:
        raise ValueError(
            f"joint_q shape {joint_q.shape} does not match robot "
            f"{robot.preset.name!r}: expected (F, {expected_cols}) for "
            f"{len(robot.dof_names())} DOFs + 7 root columns"
        )

    num_frames = joint_q.shape[0]
    if times is None:
        times = np.arange(num_frames, dtype=np.float64) / max(sample_rate, 1.0)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.shape[0] != num_frames:
        raise ValueError(
            f"times length {times.shape[0]} does not match joint_q frame "
            f"count {num_frames}"
        )

    meta = dict(meta or {})
    if include_header:
        meta.setdefault("robot", robot.preset.name)
        meta.setdefault("sample_rate", f"{sample_rate:.6f}")
        meta.setdefault("num_dofs", str(len(robot.dof_names())))

    with target.open("w", newline="", encoding="utf-8") as fp:
        if include_header:
            for key in sorted(meta):
                fp.write(f"# {key}: {meta[key]}\n")
            fp.write(",".join(header) + "\n")
        body = np.column_stack(
            [times.reshape(-1, 1), np.asarray(joint_q, dtype=np.float64)],
        )
        np.savetxt(
            fp,
            body,
            delimiter=",",
            fmt="%.6f",
            comments="",
        )
    return target


def load_robot_csv(path: str | Path) -> RobotCSV:
    """Load a robot trajectory CSV produced by :func:`save_robot_csv`.

    The reader is permissive: ``# key: value`` comments at the top are
    parsed into ``meta``; a missing header is treated as an error, but
    extra columns are simply preserved.  ``sample_rate`` is read from
    the meta line if present, otherwise inferred from the first two
    timestamps.
    """
    path = Path(path)
    header: list[str] | None = None
    meta: dict = {}
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for raw in reader:
            if not raw:
                continue
            first = raw[0]
            if first.startswith("#"):
                body = ",".join(raw).lstrip("#").strip()
                if ":" in body:
                    k, _, v = body.partition(":")
                    meta[k.strip()] = v.strip()
                continue
            if header is None:
                header = list(raw)
                continue
            rows.append(list(raw))

    if header is None:
        raise ValueError(f"{path}: no header row found")
    if header[0] != "time":
        raise ValueError(
            f"{path}: expected first column 'time', got {header[0]!r}"
        )

    dof_names = tuple(
        col[len("dof_"):] for col in header[8:] if col.startswith("dof_")
    )
    n_cols = len(header) - 1
    if not rows:
        times = np.zeros((0,), dtype=np.float64)
        joint_q = np.zeros((0, n_cols), dtype=np.float32)
    else:
        arr = np.asarray(rows, dtype=np.float64)
        times = arr[:, 0].copy()
        joint_q = arr[:, 1:].astype(np.float32, copy=False)

    sample_rate_meta = meta.get("sample_rate")
    if sample_rate_meta is not None:
        sample_rate = float(sample_rate_meta)
    elif times.shape[0] > 1:
        sample_rate = float(1.0 / max(times[1] - times[0], 1e-6))
    else:
        sample_rate = 30.0

    return RobotCSV(
        times=times,
        joint_q=joint_q,
        dof_names=dof_names,
        sample_rate=sample_rate,
        meta=meta,
    )
