# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical PARC humanoid rest skeleton for meshmimic / parc_ms MS pickles.

``dec_release`` MS pickles ship only ``motion_data`` (``root_pos`` / ``root_rot``
/ ``joint_rot``) + ``terrain_data``; they do **not** carry the skeleton, so FK
needs the rest bone offsets of the character the motion was authored against.

That character is PARC's ``data/assets/humanoid.xml`` (loaded by
``parc.anim.kin_char_model.KinCharModel.load_char_file``).  The body order is the
DFS traversal of the MJCF ``<body>`` tree, ``joint_rot[..., j-1]`` drives body
``j`` in that order, and FK is::

    body_pos[j] = body_pos[p] + quat_rotate(body_rot[p], local_translation[j])
    body_rot[j] = body_rot[p] * local_rotation[j] * joint_rot[j-1]

We bake that skeleton here as constants so import is self-contained and does not
depend on the wrong 15-bone rig (``chest`` / ``neck`` / no hands) shipped by the
earlier reference NPZ — the source of the "imported skeleton is wrong" bug.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hhtools.core.motion import Motion

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_REF = Path(__file__).resolve().parent / "data" / "parc_ms_skeleton_reference.npz"
_PARC_MS_ROOT = _REPO_ROOT / "assets/motions/meshmimic/parc_ms"

# DFS body order of PARC ``humanoid.xml`` (15 bodies, 14 non-root joints — matches
# ``joint_rot`` shape ``(T, 14, 4)`` and ``body_contacts`` shape ``(T, 15)`` in the
# dec_release pickles).  NOTE: this is the *real* PARC rig — it carries explicit
# hands and has no separate ``chest`` / ``neck`` bodies.
PARC_MS_BONE_NAMES: tuple[str, ...] = (
    "pelvis",
    "torso",
    "head",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
)

# Parent of each body in :data:`PARC_MS_BONE_NAMES` (``-1`` = root).
_PARC_MS_PARENTS: tuple[int, ...] = (-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13)

# ``pos`` attribute of each ``<body>`` in humanoid.xml — the rest bone offset in
# the parent's local frame (metres).  ``local_rotation`` is identity for every
# body (humanoid.xml declares no per-body ``quat``).
_PARC_MS_LOCAL_TRANSLATION: NDArray[np.float32] = np.asarray(
    [
        [0.0, 0.0, 0.0],            # pelvis (root)
        [0.0, 0.0, 0.236151],       # torso
        [0.0, 0.0, 0.223894],       # head
        [-0.02405, -0.18311, 0.24350],   # right_upper_arm
        [0.0, -0.274788, 0.0],      # right_lower_arm
        [0.0, -0.258947, 0.0],      # right_hand
        [-0.02405, 0.18311, 0.24350],    # left_upper_arm
        [0.0, 0.274788, 0.0],       # left_lower_arm
        [0.0, 0.258947, 0.0],       # left_hand
        [0.0, -0.084887, 0.0],      # right_thigh
        [0.0, 0.0, -0.421546],      # right_shin
        [0.0, 0.0, -0.409870],      # right_foot
        [0.0, 0.084887, 0.0],       # left_thigh
        [0.0, 0.0, -0.421546],      # left_shin
        [0.0, 0.0, -0.409870],      # left_foot
    ],
    dtype=np.float32,
)


def parc_ms_parent_indices() -> NDArray[np.int32]:
    """Parent index array for the canonical PARC humanoid 15-body rig."""
    return np.asarray(_PARC_MS_PARENTS, dtype=np.int32)


def parc_ms_local_rotation() -> NDArray[np.float32]:
    """Per-body bind rotations (identity for every PARC humanoid body)."""
    jn = len(PARC_MS_BONE_NAMES)
    local_rot = np.zeros((jn, 4), dtype=np.float32)
    local_rot[:, 3] = 1.0
    return local_rot


def build_parc_ms_skeleton_bundle_from_tpose() -> tuple[
    list[str], NDArray[np.int32], NDArray[np.float32]
]:
    """``(bone_names, parent_indices, local_translation)`` — canonical PARC rig."""
    return (
        list(PARC_MS_BONE_NAMES),
        parc_ms_parent_indices(),
        _PARC_MS_LOCAL_TRANSLATION.copy(),
    )


@lru_cache(maxsize=1)
def default_parc_ms_skeleton_bundle() -> tuple[
    list[str], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]
]:
    """``(bone_names, parent_indices, local_translation, local_rotation)``.

    These feed :func:`hhtools.io.parc_import.fk_parc_ms` directly, reproducing
    PARC ``KinCharModel.forward_kinematics`` for the dec_release rig.
    """
    return (
        list(PARC_MS_BONE_NAMES),
        parc_ms_parent_indices(),
        _PARC_MS_LOCAL_TRANSLATION.copy(),
        parc_ms_local_rotation(),
    )


def build_parc_ms_reference_motion(*, num_frames: int = 1) -> "Motion":
    """Rest-pose reference :class:`~hhtools.core.motion.Motion` (FK at identity)."""
    from hhtools.core.hierarchy import Hierarchy
    from hhtools.core.motion import Motion
    from hhtools.io.parc_import import fk_parc_ms

    bone_names, parents, local_trans, local_rot = default_parc_ms_skeleton_bundle()
    t = max(1, int(num_frames))
    root_pos = np.zeros((t, 3), dtype=np.float32)
    root_rot = np.zeros((t, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    joint_rot = np.zeros((t, len(bone_names) - 1, 4), dtype=np.float32)
    joint_rot[..., 3] = 1.0
    positions, world_q = fk_parc_ms(
        root_pos, root_rot, joint_rot, parents, local_trans, local_rot,
    )
    hierarchy = Hierarchy.from_parent_indices(bone_names, parents)
    return Motion(
        name="parc_ms_skeleton_reference",
        hierarchy=hierarchy,
        positions=positions,
        quaternions=world_q,
        framerate=30.0,
        up_axis="Z",
        source_format="npz",
        meta={"dataset": "parc_ms", "reference_rig": True},
    )


def resolve_parc_ms_reference_npz() -> Path:
    """Return a 15-body unified NPZ matching the canonical PARC rig, if present.

    The skeleton is baked in code now, so callers no longer *need* this NPZ;
    it is kept for tooling that wants an on-disk reference.  Falls back to any
    matching 15-body parc_ms clip NPZ.
    """
    if _BUNDLED_REF.is_file():
        return _BUNDLED_REF
    if _PARC_MS_ROOT.is_dir():
        for npz in sorted(_PARC_MS_ROOT.rglob("*.npz")):
            try:
                with np.load(npz, allow_pickle=False) as data:
                    if int(len(data["bone_names"])) == 15:
                        return npz
            except Exception:
                continue
    raise FileNotFoundError(
        f"parc_ms reference skeleton NPZ not found (expected {_BUNDLED_REF}). "
        "Place a 15-body parc_ms clip NPZ under assets/motions/meshmimic/."
    )


__all__ = [
    "PARC_MS_BONE_NAMES",
    "build_parc_ms_reference_motion",
    "build_parc_ms_skeleton_bundle_from_tpose",
    "default_parc_ms_skeleton_bundle",
    "parc_ms_local_rotation",
    "parc_ms_parent_indices",
    "resolve_parc_ms_reference_npz",
]
