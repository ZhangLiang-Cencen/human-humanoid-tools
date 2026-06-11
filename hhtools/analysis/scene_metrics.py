# SPDX-License-Identifier: Apache-2.0
"""Scene-aware metrics: L1 object interaction and L2 terrain awareness.

These only fire when a clip actually carries the relevant data on its
:class:`~hhtools.core.motion.Motion`:

* **L1** — ``motion.objects`` non-empty (e.g. intermimic / OMOMO): hand-object
  proximity, object motion, an ``interaction`` signal.
* **L2** — ``motion.terrain`` set (e.g. meshmimic / parc_ms / holosoma): replaces
  the flat-ground assumption with the heightfield so floating / penetration /
  foot-sliding are measured against the *local* terrain height, plus terrain
  difficulty descriptors.

Both return ``None`` when their data is absent, so flat AMASS / LAFAN clips
simply skip them and the corresponding UI fields show N/A.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.analysis.canonical import CanonicalMotionFeatures


# ----------------------------------------------------------------- L1 objects

def compute_l1_objects(
    motion, feat: CanonicalMotionFeatures, cfg: dict[str, Any]
) -> dict[str, float] | None:
    """Object-interaction metrics, or ``None`` if the clip has no objects."""
    objects = list(getattr(motion, "objects", []) or [])
    if not objects:
        return None

    dt = feat.delta_time
    obj_disp: list[float] = []
    obj_ang: list[float] = []
    centers: list[NDArray] = []  # (F, 3) per object
    for ob in objects:
        pos = np.asarray(ob.positions, dtype=np.float64)
        centers.append(pos)
        if pos.shape[0] > 1:
            obj_disp.append(float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
            quat = np.asarray(ob.quaternions, dtype=np.float64)
            dot = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
            dot = np.clip(dot, -1.0, 1.0)
            obj_ang.append(float(np.mean(2.0 * np.arccos(dot)) / max(dt, 1e-6)))

    out: dict[str, float] = {
        "num_objects": float(len(objects)),
        "object_path_length": round(max(obj_disp), 5) if obj_disp else 0.0,
        "object_ang_speed": round(max(obj_ang), 5) if obj_ang else 0.0,
    }

    # Hand-object proximity: nearest wrist to nearest object centre per frame.
    wrist_names = [n for n in ("left_wrist", "right_wrist") if feat.has(n)]
    if wrist_names and centers:
        nframes = feat.num_frames
        wrists = np.stack(
            [feat.joint_pos[n][:nframes] for n in wrist_names], axis=1
        )  # (F, W, 3)
        dists_per_obj = []
        for c in centers:
            cc = c[:nframes]
            f = min(cc.shape[0], wrists.shape[0])
            if f == 0:
                continue
            d = np.linalg.norm(wrists[:f] - cc[:f, None, :], axis=2)  # (f, W)
            dists_per_obj.append(np.min(d, axis=1))  # (f,)
        if dists_per_obj:
            min_dist = np.min(np.stack([d[: min(len(x) for x in dists_per_obj)]
                                        for d in dists_per_obj], axis=1), axis=1)
            out["hand_object_min_dist"] = round(float(np.min(min_dist)), 5)
            out["hand_object_mean_dist"] = round(float(np.mean(min_dist)), 5)
            out["hand_object_contact_ratio"] = round(
                float(np.mean(min_dist < 0.2)), 5
            )
    return out


# ----------------------------------------------------------------- L2 terrain

def compute_l2_terrain(
    motion, feat: CanonicalMotionFeatures, cfg: dict[str, Any]
) -> dict[str, float] | None:
    """Terrain descriptors + heightfield-corrected severities, or ``None``.

    The returned dict includes terrain difficulty descriptors and, when feet are
    available, corrected ``sev_floating`` / ``sev_penetration`` / ``sev_foot_slide``
    that the caller can splice back into the quality score.
    """
    terrain = getattr(motion, "terrain", None)
    if terrain is None:
        return None

    hf = np.asarray(terrain.hf, dtype=np.float64)
    out: dict[str, float] = {
        "terrain_height_range": round(float(hf.max() - hf.min()), 5),
        "terrain_height_std": round(float(hf.std()), 5),
    }
    # Roughness: mean gradient magnitude across the grid.
    if hf.shape[0] > 1 and hf.shape[1] > 1:
        gx, gy = np.gradient(hf, terrain.dx)
        out["terrain_roughness"] = round(float(np.mean(np.hypot(gx, gy))), 5)
    else:
        out["terrain_roughness"] = 0.0

    foot_names = feat.foot_joint_names()
    if not foot_names:
        return out

    th = cfg["thresholds"]
    dt = feat.delta_time
    nframes = feat.num_frames

    # Local ground under each foot via bilinear heightfield sampling.
    clearances = []  # (F,) min foot clearance above local terrain
    contacts: list[NDArray] = []
    slide_tally = 0
    slide_total = 0
    pen_depths = []
    for n in foot_names:
        pos = feat.joint_pos[n][:nframes]
        local_g = np.array(
            [terrain.height_at(float(p[0]), float(p[1])) for p in pos],
            dtype=np.float64,
        )
        clearance = pos[:, 2] - local_g
        clearances.append(clearance)
        pen_depths.append(np.clip(-clearance, 0.0, None))
        if pos.shape[0] > 1:
            vel = np.diff(pos, axis=0) / dt
            speed_xy = np.linalg.norm(vel[:, :2], axis=1)
            near = clearance[1:] < float(th["contact_height_m"])
            slide_total += int(np.sum(near))
            slide_tally += int(np.sum(near & (speed_xy > float(th["foot_slide_speed_mps"]))))

    clear_stack = np.stack(clearances, axis=1)  # (F, K)
    min_clear = np.min(clear_stack, axis=1)
    out["foot_terrain_clearance_mean"] = round(float(np.mean(min_clear)), 5)

    # Corrected severities (override flat-ground estimates).
    out["sev_floating"] = round(
        float(np.mean(min_clear > float(th["floating_height_m"]))), 5
    )
    pen = np.concatenate(pen_depths) if pen_depths else np.zeros(1)
    out["sev_penetration"] = round(float(min(np.mean(pen) / 0.1, 1.0)), 5)
    out["sev_foot_slide"] = round(
        float(slide_tally / slide_total) if slide_total else 0.0, 5
    )
    return out


__all__ = ["compute_l1_objects", "compute_l2_terrain"]
