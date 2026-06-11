"""Scaled skeleton preview for the web UI (matches Viser ``_compute_scaled_preview``).

The yellow overlay uses **uniform** ``robot_height / human_height`` scaling on the
motion's source topology when the clip has enough bones (>= 10).  The older
``NewtonBasicPipeline.scale_only`` path only exposes IK canonical effectors and
distorts dense rigs (OMOMO, meshmimic, terrain clips).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hhtools.core.grounding import human_source_floor_z_world
from hhtools.core.motion import Motion
from hhtools.viewer.anatomy import (
    exclude_joint_from_compact_scaled_preview,
    exclude_unmapped_head_neck_from_scaled_preview,
    motion_has_interaction_scene,
    scaled_overlay_exclude_bone_indices,
)


def resolve_web_scaler_config(
    model,
    motion: Motion,
    reference: str,
    human_height: float,
):
    """Same scaler resolution as :func:`_retarget_single` (bundled > calibration)."""

    from hhtools.retarget.calibration import load_calibration, resolve_calibration_file
    from hhtools.robot.retarget_profile import (
        bundled_scaler_path,
        resolve_retarget_scaler_config,
    )

    preset = model.preset
    cal_path = resolve_calibration_file(preset.urdf_path.parent, reference)
    if cal_path is None and bundled_scaler_path(preset, reference) is None:
        raise ValueError(
            f"robot {preset.name!r} has no bundled scaler or calibration "
            f"for reference {reference!r}"
        )
    calibration = load_calibration(cal_path) if cal_path is not None else None
    return resolve_retarget_scaler_config(
        preset,
        reference,
        calibration=calibration,
        model=model,
        motion=motion,
        human_height=float(human_height),
    )


def _scaler_skeleton_segment_indices(
    joint_names: tuple[str, ...] | list[str],
    hierarchy,
    *,
    ik_map_canonicals: frozenset[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Edges (parent→child) in scaler joint index space (Viser copy)."""
    pi = np.asarray(hierarchy.parent_indices, dtype=np.int64)
    hnames = list(hierarchy.bone_names)
    h_idx = {hnames[i]: i for i in range(len(hnames))}
    name_to_i = {n: i for i, n in enumerate(joint_names)}
    src: list[int] = []
    dst: list[int] = []
    for i, n in enumerate(joint_names):
        if n not in h_idx:
            continue
        if str(n).lower().endswith("footmod"):
            continue
        hi = int(h_idx[n])
        p = int(pi[hi])
        anc_sc = None
        while p >= 0:
            pn = hnames[p]
            if pn in name_to_i:
                j = int(name_to_i[pn])
                if j != i:
                    anc_sc = j
                break
            p = int(pi[p])
        if anc_sc is not None:
            src.append(anc_sc)
            dst.append(i)
    fs: list[int] = []
    fd: list[int] = []
    for s, d in zip(src, dst, strict=True):
        ns, nd = joint_names[s], joint_names[d]
        if exclude_joint_from_compact_scaled_preview(ns) or exclude_joint_from_compact_scaled_preview(nd):
            continue
        if ik_map_canonicals and (
            exclude_unmapped_head_neck_from_scaled_preview(ns, ik_map_canonicals=ik_map_canonicals)
            or exclude_unmapped_head_neck_from_scaled_preview(nd, ik_map_canonicals=ik_map_canonicals)
        ):
            continue
        fs.append(s)
        fd.append(d)
    return (
        np.asarray(fs, dtype=np.int32),
        np.asarray(fd, dtype=np.int32),
    )


def _visible_joint_indices(
    joint_names: list[str],
    ik_canons: frozenset[str],
) -> np.ndarray:
    idx = []
    for i, n in enumerate(joint_names):
        if exclude_joint_from_compact_scaled_preview(n):
            continue
        if ik_canons and exclude_unmapped_head_neck_from_scaled_preview(
            n, ik_map_canonicals=ik_canons,
        ):
            continue
        idx.append(i)
    return np.asarray(idx, dtype=np.int32)


def _uniform_overlay_z_correction(
    motion: Motion,
    scaler,
    ratio: float,
) -> float:
    """Pelvis-height delta: soma-style IK scaler minus uniform yellow overlay.

    Per-joint ``scaler.apply`` adds ``root_z_offset`` so frame-0 IK lands on the
    calibrated robot; the uniform overlay omits that shift, leaving a constant
    ~10–20 cm vertical gap vs the retargeted robot.  Measured on the anatomical
    root (``Hips`` for SOMA) at frame 0 — constant across frames for a given clip.
    """
    root_name = str(scaler.config.root_joint)
    try:
        j_root = list(scaler.joint_names).index(root_name)
    except ValueError:
        return 0.0
    bone_names = motion.hierarchy.bone_names
    if root_name not in bone_names:
        return 0.0
    hi = bone_names.index(root_name)
    z_floor = float(human_source_floor_z_world(motion))
    uniform_z = float((motion.positions[0, hi, 2] - z_floor) * ratio)
    # Only frame 0 is read below, so scale a 1-frame slice instead of the whole
    # clip — ``scaler.apply`` over thousands of frames was a large, pure waste
    # in the post-retarget serialisation path.
    import dataclasses

    motion_f0 = dataclasses.replace(
        motion,
        positions=motion.positions[:1],
        quaternions=motion.quaternions[:1],
    )
    eff = scaler.apply(motion_f0)
    scaler_z = float(eff.transforms[0, j_root, 2])
    return scaler_z - uniform_z


def resolve_scaled_overlay_z_correction(
    motion: Motion,
    scaler,
    ratio: float,
) -> float:
    """Vertical shift for the yellow skeleton overlay vs uniform scaling alone.

    Clips with terrain and/or interaction props must **not** apply this correction:
    the overlay must stay co-aligned with the uniformly-scaled scene geometry
    (``scaled_scene`` terrain / objects use the same ``z_min`` + ``ratio`` chain
    as :meth:`InteractionMeshPipeline._build_scaled_source_pose`).  Adding
    ``root_z_offset``-style correction to the skeleton only lifts it off the
    scaled heightfield — the bug reported on ``parc_ms`` terrain clips where the
    retargeted robot (solver frame) looks fine but the yellow overlay floats.
    """
    if motion_has_interaction_scene(motion):
        return 0.0
    return _uniform_overlay_z_correction(motion, scaler, ratio)


def _uniform_scaled_joint_positions(
    motion: Motion,
    scaler_cfg,
    human_height: float,
    joint_names: tuple[str, ...] | list[str],
    *,
    ik_canons: frozenset[str],
    z_correction: float = 0.0,
) -> np.ndarray:
    """Uniform ``robot_height / human_height`` positions for yellow overlay joints.

    Matches Viser ``_compute_scaled_preview``: dense rigs (holosoma, OMOMO,
    meshmimic, SMPL) keep source bone proportions; per-joint soma-style
    ``scaler.apply`` targets are only exact at calibration rest and distort
    limb lengths on motion frames (elongated arms, puffed torso).

    ``z_correction`` (from :func:`_uniform_overlay_z_correction`) vertically
    aligns the overlay with the foot-grounded IK / retargeted robot.
    """
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion

    ratio = float(
        uniform_overlay_scale_for_motion(
            scaler_cfg, human_height, motion, ik_map_keys=ik_canons,
        )
    )
    z_min = float(human_source_floor_z_world(motion))
    src_pos = np.asarray(motion.positions, dtype=np.float32).copy()
    src_pos[:, :, 2] -= z_min
    src_pos *= ratio
    if abs(z_correction) > 1e-6:
        src_pos[:, :, 2] += np.float32(z_correction)

    hname_to_idx = {n: i for i, n in enumerate(motion.hierarchy.bone_names)}
    parents_h = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    root_arr = np.where(parents_h < 0)[0]
    root_idx = int(root_arr[0]) if root_arr.size > 0 else 0
    mapped = np.asarray(
        [hname_to_idx.get(n, root_idx) for n in joint_names],
        dtype=np.int64,
    )
    return src_pos[:, mapped, :].astype(np.float32, copy=False)


def _uniform_scaled_preview_fallback(
    motion: Motion,
    scaler_cfg,
    human_height: float,
    ik_canons: frozenset[str],
    *,
    max_frames: int = 0,
    z_correction: float = 0.0,
) -> dict[str, Any]:
    """Numpy-only scaled overlay when ``newton`` is not installed."""
    from hhtools.web.serialize import _downsample_indices

    jn = list(motion.hierarchy.bone_names)
    pos = _uniform_scaled_joint_positions(
        motion, scaler_cfg, human_height, jn,
        ik_canons=ik_canons, z_correction=z_correction,
    )
    idx = _downsample_indices(int(pos.shape[0]), max_frames, motion=motion)
    pos = pos[idx]
    parents = motion.hierarchy.parent_indices.tolist()
    return {
        "name": "scaled_targets",
        "bone_names": jn,
        "parent_indices": parents,
        "frame_indices": idx.tolist(),
        "positions": np.round(pos, 4).tolist(),
        "num_frames_total": int(motion.num_frames),
        "framerate": float(motion.framerate),
        "duration": float(motion.duration),
    }


def compute_web_scaled_preview(
    model,
    motion: Motion,
    reference: str,
    human_height: float,
    *,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Build scaled skeleton payload for the browser (dense topology when possible)."""
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
    from hhtools.web.serialize import _downsample_indices, serialize_scaled_preview

    scaler_cfg = resolve_web_scaler_config(
        model, motion, reference, float(human_height),
    )
    ik_canons = frozenset(model.preset.ik_map.keys()) if model.preset.ik_map else frozenset()

    if int(motion.num_bones) < 10:
        try:
            from hhtools.retarget.newton_basic import NewtonBasicPipeline, PipelineConfig

            pipeline = NewtonBasicPipeline(
                model,
                scaler_config=scaler_cfg,
                pipeline_config=PipelineConfig(ik_iterations=1),
                human_height=float(human_height),
                configure_warp=False,
            )
            return serialize_scaled_preview(
                pipeline.scale_only(motion),
                max_frames=max_frames,
                ik_map_canonicals=ik_canons,
            )
        except ModuleNotFoundError:
            return _uniform_scaled_preview_fallback(
                motion, scaler_cfg, float(human_height), ik_canons, max_frames=max_frames,
            )

    scaler = HumanToRobotScaler(
        motion.hierarchy, scaler_cfg, human_height=float(human_height),
    )
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion

    overlay_ratio = float(
        uniform_overlay_scale_for_motion(
            scaler_cfg, float(human_height), motion, ik_map_keys=ik_canons,
        )
    )
    z_correction = resolve_scaled_overlay_z_correction(motion, scaler, overlay_ratio)
    jn = list(scaler.joint_names)
    seg_s, seg_d = _scaler_skeleton_segment_indices(jn, motion.hierarchy, ik_map_canonicals=ik_canons)
    if int(seg_s.size) == 0:
        try:
            from hhtools.retarget.newton_basic import NewtonBasicPipeline, PipelineConfig

            pipeline = NewtonBasicPipeline(
                model,
                scaler_config=scaler_cfg,
                pipeline_config=PipelineConfig(ik_iterations=1),
                human_height=float(human_height),
                configure_warp=False,
            )
            return serialize_scaled_preview(
                pipeline.scale_only(motion),
                max_frames=max_frames,
                ik_map_canonicals=ik_canons,
            )
        except ModuleNotFoundError:
            return _uniform_scaled_preview_fallback(
                motion, scaler_cfg, float(human_height), ik_canons,
                max_frames=max_frames, z_correction=z_correction,
            )

    # Dense source-topology overlay: uniform ``robot_height / human_height``
    # scaling on raw joint positions (Viser parity).  IK still consumes
    # per-joint ``scaler.apply`` targets; only the yellow *display* stays
    # proportionally faithful to the source skeleton.
    pos_m = _uniform_scaled_joint_positions(
        motion, scaler_cfg, float(human_height), jn,
        ik_canons=ik_canons, z_correction=z_correction,
    )

    vis_idx = _visible_joint_indices(jn, ik_canons)
    if vis_idx.size == 0:
        vis_idx = np.arange(len(jn), dtype=np.int32)

    # Remap segments to visible joint subset.
    old_to_new = {int(old): int(new) for new, old in enumerate(vis_idx.tolist())}
    bone_names = [jn[int(i)] for i in vis_idx.tolist()]
    seg_pairs: list[tuple[int, int]] = []
    overlay_exclude = (
        set()
        if motion_has_interaction_scene(motion)
        else scaled_overlay_exclude_bone_indices(motion, ik_canons)
    )
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    src2can = auto_source_to_canonical(tuple(motion.hierarchy.bone_names))
    can_to_hidx: dict[str, int] = {}
    for i, raw in enumerate(motion.hierarchy.bone_names):
        can_to_hidx[str(src2can.get(raw, raw)).lower()] = i
    for s, d in zip(seg_s.tolist(), seg_d.tolist(), strict=True):
        if int(s) not in old_to_new or int(d) not in old_to_new:
            continue
        hd = str(jn[int(d)]).lower()
        hi_d = can_to_hidx.get(hd)
        if hi_d is not None and int(hi_d) in overlay_exclude:
            continue
        seg_pairs.append((old_to_new[int(s)], old_to_new[int(d)]))
    parent_indices = [-1] * len(bone_names)
    for s, d in seg_pairs:
        parent_indices[int(d)] = int(s)

    positions = pos_m[:, vis_idx, :]
    idx = _downsample_indices(int(positions.shape[0]), max_frames, motion=motion)
    positions = positions[idx]

    return {
        "name": "scaled_targets",
        "bone_names": bone_names,
        "parent_indices": parent_indices,
        "frame_indices": idx.tolist(),
        "positions": np.round(positions, 4).tolist(),
        "num_frames_total": int(motion.num_frames),
        "framerate": float(motion.framerate),
        "duration": float(motion.duration),
    }
