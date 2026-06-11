# SPDX-License-Identifier: Apache-2.0
"""L0 skeleton-kinematics metrics + physics-feasibility quality score.

Everything here is computed from a :class:`~hhtools.analysis.canonical.CanonicalMotionFeatures`
so it is identical for human source clips and retargeted robot trajectories.

Two metric families are produced:

* **Dynamics / complexity** — kinetic & acceleration energy, root speed, COM
  height variation, locomotion stats, leg/arm energy, inverted ratio.  Used to
  separate static vs high-dynamic vs flip/jump behaviours.
* **Quality (``S_phy``)** — interpretable, *flat-ground* approximations of the
  six LIMMT penalty modes (floating, penetration, velocity violation, foot
  sliding, self-collision, jerk).  Terrain-aware corrections live in
  :mod:`hhtools.analysis.scene_metrics` and override the relevant terms when a
  clip carries a heightfield.

These are kinematic approximations (no rigid-body simulator), surfaced so the
user can rank and triage; they are not a substitute for a full sim replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.analysis.canonical import CanonicalMotionFeatures


def _rotate_vec_by_quat(quat: NDArray, vec: NDArray) -> NDArray:
    """Rotate ``vec`` (3,) by an array of xyzw quaternions ``(F, 4)`` -> ``(F, 3)``."""
    q = np.asarray(quat, dtype=np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    # t = 2 * cross(q_xyz, v)
    qxyz = np.stack([x, y, z], axis=1)
    t = 2.0 * np.cross(qxyz, v[None, :])
    rotated = v[None, :] + w[:, None] * t + np.cross(qxyz, t)
    return rotated.astype(np.float32)


def _quat_angular_speed(quat: NDArray, dt: float) -> NDArray:
    """Per-frame angular speed magnitude (rad/s) from consecutive xyzw quats."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[0] < 2:
        return np.zeros((q.shape[0],), dtype=np.float32)
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    ang = 2.0 * np.arccos(dot) / max(dt, 1e-6)
    out = np.empty((q.shape[0],), dtype=np.float32)
    out[1:] = ang.astype(np.float32)
    out[0] = out[1] if q.shape[0] > 1 else 0.0
    return out


def _safe_stat(values: NDArray, fn) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(fn(arr))


def _ground_z(feat: CanonicalMotionFeatures) -> float:
    """Estimate flat-ground height as a low percentile of foot heights."""
    foot_names = feat.foot_joint_names()
    if not foot_names:
        return float(np.percentile(feat.root_pos[:, 2], 2)) if feat.num_frames else 0.0
    zs = np.concatenate([feat.joint_pos[n][:, 2] for n in foot_names])
    zs = zs[np.isfinite(zs)]
    if zs.size == 0:
        return 0.0
    return float(np.percentile(zs, 2.0))


def compute_dynamics(feat: CanonicalMotionFeatures, cfg: dict[str, Any]) -> dict[str, float]:
    """Dynamics / complexity metrics (all source-agnostic)."""
    dt = feat.delta_time
    th = cfg["thresholds"]
    lam = float(cfg["complexity"]["lambda_accel"])

    out: dict[str, float] = {
        "duration_s": round(feat.duration, 4),
        "fps": round(feat.fps, 3),
        "num_frames": int(feat.num_frames),
    }

    if feat.num_frames < 2:
        out.update({
            "joint_kinetic_energy": 0.0,
            "joint_accel_energy": 0.0,
            "complexity": 0.0,
            "root_speed_xy": 0.0,
            "root_speed_xy_p95": 0.0,
            "root_speed_z": 0.0,
            "root_turn_rate": 0.0,
            "com_height_std": 0.0,
            "com_height_range": 0.0,
            "airborne_ratio": 0.0,
            "path_efficiency": 1.0,
            "step_count": 0.0,
            "step_freq": 0.0,
            "leg_energy": 0.0,
            "arm_energy": 0.0,
            "inverted_ratio": 0.0,
            "max_torso_tilt_deg": 0.0,
        })
        return out

    # --- joint kinetic / acceleration energy (LIMMT C(x)) ---------------------
    # Prefer robot DOF velocities (closer to the paper's q̇); else use canonical
    # joint linear velocities as a proxy.
    if feat.dof_q is not None and feat.dof_q.shape[1] > 0:
        qd = np.diff(feat.dof_q, axis=0) / dt
        qdd = np.diff(qd, axis=0) / dt if qd.shape[0] > 1 else np.zeros_like(qd)
        kin = float(np.mean(np.sum(qd**2, axis=1))) if qd.size else 0.0
        acc = float(np.mean(np.sum(qdd**2, axis=1))) if qdd.size else 0.0
    else:
        jp = feat.all_joint_positions()  # (F, J, 3)
        vel = np.diff(jp, axis=0) / dt
        accel = np.diff(vel, axis=0) / dt if vel.shape[0] > 1 else np.zeros_like(vel)
        kin = float(np.mean(np.sum(vel**2, axis=(1, 2)))) if vel.size else 0.0
        acc = float(np.mean(np.sum(accel**2, axis=(1, 2)))) if accel.size else 0.0

    out["joint_kinetic_energy"] = round(kin, 5)
    out["joint_accel_energy"] = round(acc, 5)
    out["complexity"] = round(kin + lam * acc, 5)

    # --- root motion ----------------------------------------------------------
    root_vel = np.diff(feat.root_pos, axis=0) / dt  # (F-1, 3)
    speed_xy = np.linalg.norm(root_vel[:, :2], axis=1)
    out["root_speed_xy"] = round(_safe_stat(speed_xy, np.mean), 5)
    out["root_speed_xy_p95"] = round(_safe_stat(speed_xy, lambda a: np.percentile(a, 95)), 5)
    out["root_speed_z"] = round(_safe_stat(np.abs(root_vel[:, 2]), np.mean), 5)
    out["root_turn_rate"] = round(_safe_stat(_quat_angular_speed(feat.root_quat, dt), np.mean), 5)

    # --- COM height (use root z as a cheap proxy) -----------------------------
    com_z = feat.root_pos[:, 2]
    out["com_height_std"] = round(_safe_stat(com_z, np.std), 5)
    out["com_height_range"] = round(
        _safe_stat(com_z, np.max) - _safe_stat(com_z, np.min), 5
    )

    # --- airborne / contact / steps ------------------------------------------
    ground = _ground_z(feat)
    foot_names = feat.foot_joint_names()
    if foot_names:
        foot_z = np.stack([feat.joint_pos[n][:, 2] for n in foot_names], axis=1)  # (F, K)
        min_foot_above = np.min(foot_z, axis=1) - ground
        airborne = min_foot_above > float(th["floating_height_m"])
        out["airborne_ratio"] = round(float(np.mean(airborne)), 5)

        # contact = any foot near ground; count rising edges -> steps
        contact_each = (foot_z - ground) < float(th["contact_height_m"])
        steps = 0
        for k in range(contact_each.shape[1]):
            c = contact_each[:, k].astype(np.int8)
            steps += int(np.sum((c[1:] == 1) & (c[:-1] == 0)))
        out["step_count"] = float(steps)
        out["step_freq"] = round(steps / max(feat.duration, 1e-6), 4)
    else:
        out["airborne_ratio"] = 0.0
        out["step_count"] = 0.0
        out["step_freq"] = 0.0

    # --- path efficiency ------------------------------------------------------
    path_len = float(np.sum(np.linalg.norm(np.diff(feat.root_pos[:, :2], axis=0), axis=1)))
    disp = float(np.linalg.norm(feat.root_pos[-1, :2] - feat.root_pos[0, :2]))
    out["path_efficiency"] = round(disp / path_len, 4) if path_len > 1e-6 else 1.0

    # --- leg / arm energy -----------------------------------------------------
    out["leg_energy"] = round(_group_energy(feat, feat.leg_joint_names(), dt), 5)
    out["arm_energy"] = round(_group_energy(feat, feat.arm_joint_names(), dt), 5)

    # --- inverted / torso tilt ------------------------------------------------
    up_body = _rotate_vec_by_quat(feat.root_quat, np.array([0.0, 0.0, 1.0]))
    cos_ang = np.clip(up_body[:, 2], -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_ang))
    inv_thresh = float(th["inverted_angle_deg"])
    out["inverted_ratio"] = round(float(np.mean(tilt_deg > inv_thresh)), 5)
    out["max_torso_tilt_deg"] = round(_safe_stat(tilt_deg, np.max), 3)

    return out


def _group_energy(feat: CanonicalMotionFeatures, names: list[str], dt: float) -> float:
    if not names:
        return 0.0
    vels = []
    for n in names:
        v = feat.joint_velocity(n)
        if v is not None:
            vels.append(np.sum(v**2, axis=1))
    if not vels:
        return 0.0
    return float(np.mean(np.sum(np.stack(vels, axis=1), axis=1)))


def compute_quality(feat: CanonicalMotionFeatures, cfg: dict[str, Any]) -> dict[str, float]:
    """Flat-ground physics-feasibility approximation (LIMMT-style ``S_phy``).

    Returns the six normalised severities (``sev_*`` in [0, 1]) and the composite
    ``s_phy`` in [0, 100].  ``scene_metrics`` may overwrite ``sev_floating`` /
    ``sev_penetration`` / ``sev_foot_slide`` and recompute ``s_phy`` when a clip
    carries terrain.
    """
    th = cfg["thresholds"]
    sev = _quality_severities(feat, th)
    return finalize_quality(sev, cfg)


def _quality_severities(feat: CanonicalMotionFeatures, th: dict[str, Any]) -> dict[str, float]:
    dt = feat.delta_time
    sev = {
        "sev_floating": 0.0,
        "sev_penetration": 0.0,
        "sev_vel_violation": 0.0,
        "sev_foot_slide": 0.0,
        "sev_self_collision": 0.0,
        "sev_jerk": 0.0,
    }
    if feat.num_frames < 3:
        return sev

    ground = _ground_z(feat)
    foot_names = feat.foot_joint_names()
    if foot_names:
        foot_z = np.stack([feat.joint_pos[n][:, 2] for n in foot_names], axis=1)
        min_above = np.min(foot_z, axis=1) - ground
        # Floating: sustained whole-body airborne. Severity = airborne fraction.
        sev["sev_floating"] = float(np.mean(min_above > float(th["floating_height_m"])))
        # Penetration: feet below ground. Severity = clipped mean depth / 0.1 m.
        depth = np.clip(ground - foot_z, 0.0, None)
        sev["sev_penetration"] = float(min(np.mean(depth) / 0.1, 1.0))
        # Foot slide: horizontal foot speed while near ground.
        slide_frac = _foot_slide_fraction(feat, foot_names, ground, th, dt)
        sev["sev_foot_slide"] = slide_frac

    # Velocity violation: fraction of frames any joint linear speed is implausible
    # (> 12 m/s end-effector). Robot path uses DOF velocity z-score instead.
    sev["sev_vel_violation"] = _vel_violation_severity(feat, dt)

    # Jerk: normalised mean root jerk magnitude.
    sev["sev_jerk"] = _jerk_severity(feat, dt)

    # Self-collision: not computed without meshes (left 0; lowest weight anyway).
    return sev


def _foot_slide_fraction(
    feat: CanonicalMotionFeatures, foot_names: list[str], ground: float,
    th: dict[str, Any], dt: float,
) -> float:
    contact_h = float(th["contact_height_m"])
    slide_v = float(th["foot_slide_speed_mps"])
    tally = 0
    total = 0
    for n in foot_names:
        pos = feat.joint_pos[n]
        vel = np.diff(pos, axis=0) / dt
        speed_xy = np.linalg.norm(vel[:, :2], axis=1)
        near = (pos[1:, 2] - ground) < contact_h
        total += int(np.sum(near))
        tally += int(np.sum(near & (speed_xy > slide_v)))
    if total == 0:
        return 0.0
    return float(tally / total)


def _vel_violation_severity(feat: CanonicalMotionFeatures, dt: float) -> float:
    if feat.dof_q is not None and feat.dof_q.shape[1] > 0:
        qd = np.abs(np.diff(feat.dof_q, axis=0) / dt)
        if qd.size == 0:
            return 0.0
        # rad/s; flag frames exceeding a generous 20 rad/s limit.
        frac = float(np.mean(np.any(qd > 20.0, axis=1)))
        return min(frac * 2.0, 1.0)
    jp = feat.all_joint_positions()
    vel = np.linalg.norm(np.diff(jp, axis=0) / dt, axis=2)  # (F-1, J)
    if vel.size == 0:
        return 0.0
    frac = float(np.mean(np.any(vel > 12.0, axis=1)))
    return min(frac * 2.0, 1.0)


def _jerk_severity(feat: CanonicalMotionFeatures, dt: float) -> float:
    p = feat.root_pos
    if p.shape[0] < 4:
        return 0.0
    jerk = np.diff(p, n=3, axis=0) / (dt**3)
    mag = np.linalg.norm(jerk, axis=1)
    # Normalise: 5000 m/s^3 maps to severity ~1.
    return float(min(_safe_stat(mag, np.mean) / 5000.0, 1.0))


def finalize_quality(sev: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    """Combine severities into ``s_phy`` using the configured weights."""
    weights = cfg["quality"]["weights"]
    deduction = (
        weights["floating"] * sev["sev_floating"]
        + weights["penetration"] * sev["sev_penetration"]
        + weights["vel_violation"] * sev["sev_vel_violation"]
        + weights["foot_slide"] * sev["sev_foot_slide"]
        + weights["self_collision"] * sev["sev_self_collision"]
        + weights["jerk"] * sev["sev_jerk"]
    )
    s_phy = float(np.clip(100.0 - deduction, 0.0, 100.0))
    out = {k: round(v, 5) for k, v in sev.items()}
    out["s_phy"] = round(s_phy, 3)
    return out


__all__ = ["compute_dynamics", "compute_quality", "finalize_quality"]
