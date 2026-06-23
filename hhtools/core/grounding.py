# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Foot-based floor detection and clip-wide skeleton grounding.

The **split** vertical convention (foot floor for the skeleton, a separate
``z_offset`` for the terrain heightfield so low cells are not buried) is enabled
for ``20260429_mocap`` and ``parc_ms`` meshmimic clips.  Interaction-mesh
retarget uses :func:`human_source_floor_z_world` for the skeleton floor (clip-wide
minimum over all joints); split rules only affect terrain ``z_offset`` when the
heightfield minimum sits below that plane.

See :func:`use_split_terrain_grounding`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hhtools.core.motion import Motion

# Meshmimic MoCap export tree (see ``hhtools.viewer.library``).
_SPLIT_GROUNDING_FOLDER_LABEL = "20260429_mocap"

_FOOT_NAME_RE = re.compile(
    r"(foot|toe|ankle)$|_(foot|toe|ankle)$|(foot|toe|ankle)[._]",
    re.IGNORECASE,
)


def foot_contact_bone_indices(bone_names: tuple[str, ...]) -> NDArray[np.int64]:
    """Indices of bones likely at ground contact (feet / toes / ankles).

    Excludes false positives like ``ForeArm``.  When at least two indices exist,
    callers typically take ``min(Z)`` over those bones across all frames as a
    foot floor — Mixamo / FBX clips often leave ankle joints visibly floating if
    the global minimum came from fingertips or props.

    Prefer :func:`preferred_floor_contact_bone_indices` for floor height — it
    drops ``*FootMod`` markers and prefers ankle hubs over toe/end bones.
    """

    idx: list[int] = []
    for i, raw in enumerate(bone_names):
        n = raw.lower().replace("mixamorig:", "").replace("mixamo:", "")
        if "forearm" in n:
            continue
        if _FOOT_NAME_RE.search(n) or (
            "foot" in n and "root" not in n
        ) or ("toe" in n and "finger" not in n):
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def preferred_floor_contact_bone_indices(
    bone_names: tuple[str, ...],
) -> NDArray[np.int64]:
    """Foot indices that define the human floor plane for grounding / preview.

    meshmimic / holosoma ``*FootMod`` sole markers are auxiliary Laplacian
    anchors — they can sit above the parent foot or carry bad Z in some
    exports.  Including them in ``min(Z)`` shifts the whole actor up and
    leaves the yellow scaled skeleton floating above the robot.

    When canonical ``left_ankle`` / ``right_ankle`` bones are identifiable,
    prefer them over toe/end bones so the floor matches the ankle joints
    drawn in the overlay (toe joints dipping below the floor in mocap would
    otherwise lift the whole body).
    """

    raw = foot_contact_bone_indices(bone_names)
    if raw.size == 0:
        return raw

    kept: list[int] = []
    for i in raw:
        n = bone_names[int(i)].lower()
        if "footmod" in n:
            continue
        kept.append(int(i))

    if len(kept) < 2:
        if kept:
            return np.asarray(kept, dtype=np.int64)
        return raw

    try:
        from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

        src2can = auto_source_to_canonical(bone_names)
    except Exception:
        src2can = {}

    ankle: list[int] = []
    for i in kept:
        canon = str(src2can.get(bone_names[i], "")).lower()
        if canon in ("left_ankle", "right_ankle"):
            ankle.append(i)
    if len(ankle) >= 2:
        return np.asarray(ankle, dtype=np.int64)

    hub: list[int] = []
    for i in kept:
        bn = bone_names[i].lower().replace("mixamorig:", "").replace("mixamo:", "")
        if bn.endswith("foot") and "toe" not in bn and "ball" not in bn:
            hub.append(i)
    if len(hub) >= 2:
        return np.asarray(hub, dtype=np.int64)

    return np.asarray(kept, dtype=np.int64)


def clip_floor_z_in_positions(positions: NDArray) -> float:
    """Minimum world Z over all joints in ``positions`` (``(F, J, 3)`` or ``(J, 3)``)."""

    pos = np.asarray(positions, dtype=np.float32)
    if pos.size == 0:
        return 0.0
    if pos.ndim == 2:
        return float(pos[:, 2].min())
    return float(pos[:, :, 2].min())


def foot_floor_z_in_positions(
    positions: NDArray,
    bone_names: tuple[str, ...],
) -> float:
    """Minimum world Z over preferred floor-contact bones in ``positions``."""

    pos = np.asarray(positions, dtype=np.float32)
    if pos.size == 0:
        return 0.0
    foot_i = preferred_floor_contact_bone_indices(bone_names)
    if foot_i.size >= 2:
        if pos.ndim == 2:
            return float(pos[foot_i, 2].min())
        return float(pos[:, foot_i, 2].min())
    if pos.ndim == 2:
        return float(pos[:, 2].min())
    return float(pos[:, :, 2].min())


def use_split_terrain_grounding(motion: "Motion") -> bool:
    """Whether to use foot-floor + separate terrain ``z_offset``.

    Enabled for ``20260429_mocap`` and ``parc_ms`` clips (heightfield terrain whose
    ``min(hf)`` can sit below the foot floor — a single ``dz`` for skeleton + HF
    leaves terrain half-buried in the viewer grid).
    """

    meta = getattr(motion, "meta", None)
    if not isinstance(meta, dict):
        return False
    if meta.get("dataset") in ("parc_ms",):
        return True
    if meta.get("library_folder_label") in (_SPLIT_GROUNDING_FOLDER_LABEL, "parc_ms"):
        return True
    if meta.get("split_terrain_grounding") is True:
        return True
    for key in ("mocap_source_take_dir", "npz_path", "source_npz_path", "source_pkl"):
        raw = meta.get(key)
        if not isinstance(raw, str):
            continue
        norm = raw.replace("\\", "/")
        if f"meshmimic/{_SPLIT_GROUNDING_FOLDER_LABEL}/" in norm:
            return True
        if f"/{_SPLIT_GROUNDING_FOLDER_LABEL}/" in norm:
            return True
        if "meshmimic/parc_ms/" in norm:
            return True
    return False


def human_source_floor_z_world(motion: "Motion") -> float:
    """Minimum world Z over all joints across the full clip.

    Uses the clip-wide lowest point (toes, knees in kneeling, hands on floor,
    etc.) so poses where feet leave the ground still normalize correctly.
    """

    return clip_floor_z_in_positions(np.asarray(motion.positions, dtype=np.float32))


def terrain_heightfield_z_offset_world(motion: "Motion", z_human_floor_m: float) -> float:
    """``z_offset`` for :meth:`TerrainHeightfield.scaled` when split grounding is on.

    When :func:`use_split_terrain_grounding` is false, callers should pass the same
    ``z_human_floor_m`` used for the skeleton (legacy single ``z_min``).

    When true and a heightfield exists, returns ``min(z_human_floor_m, min(hf))``
    so obstacle geometry below the foot floor still normalises with its lowest
    sample at the working plane.
    """

    if not use_split_terrain_grounding(motion):
        return float(z_human_floor_m)
    terr = getattr(motion, "terrain", None)
    if terr is None:
        return float(z_human_floor_m)
    # ``parc_ms`` motion + terrain live in one shared world frame (the actor is
    # authored standing/stepping on the heightfield).  The terrain therefore
    # MUST use the *same* Z reference as the skeleton, otherwise the surface
    # detaches from the feet it contacts.  Only datasets whose terrain is
    # grounded independently (``20260429_mocap``) take the ``min(hf)`` floor.
    meta = getattr(motion, "meta", None)
    if isinstance(meta, dict) and meta.get("dataset") == "parc_ms":
        return float(z_human_floor_m)
    return min(float(z_human_floor_m), float(np.min(terr.hf)))


__all__ = [
    "clip_floor_z_in_positions",
    "foot_contact_bone_indices",
    "foot_floor_z_in_positions",
    "human_source_floor_z_world",
    "preferred_floor_contact_bone_indices",
    "terrain_heightfield_z_offset_world",
    "use_split_terrain_grounding",
]
