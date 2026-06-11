# SPDX-License-Identifier: Apache-2.0
"""Rule-based, interpretable tags.

Tags are *non-destructive labels*, never filters.  "Unreasonable" data is just
another tag (``quality_bad``); the user decides whether to keep or drop it.

Two phases:

* :func:`assign_clip_tags` — absolute, per-clip tags that depend only on that
  clip's metrics (quality band, locomotion family, inverted/flip, jump,
  has_object/has_terrain, interaction).
* :func:`assign_dataset_tags` — relative tags that need the whole collection's
  distribution (dynamics quartiles -> static/low/mid/high/burst, leg_static).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def assign_clip_tags(metrics: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Absolute per-clip tags from one clip's metric dict."""
    th = cfg["thresholds"]
    q = cfg["quality"]
    tags: list[str] = []

    # --- quality band ---------------------------------------------------------
    s_phy = float(metrics.get("s_phy", 100.0))
    if s_phy < float(q["bad_below"]):
        tags.append("quality_bad")
    elif s_phy < float(q["warn_below"]):
        tags.append("quality_warn")
    else:
        tags.append("quality_ok")

    # --- locomotion family ----------------------------------------------------
    root_speed = float(metrics.get("root_speed_xy", 0.0))
    airborne = float(metrics.get("airborne_ratio", 0.0))
    inverted = float(metrics.get("inverted_ratio", 0.0))
    turn_rate = float(metrics.get("root_turn_rate", 0.0))

    if inverted > 0.05 or float(metrics.get("max_torso_tilt_deg", 0.0)) > 120.0:
        tags.append("flip")
    if airborne > float(th["jump_airborne_ratio"]):
        tags.append("jump")
    if root_speed < float(th["static_root_speed_mps"]):
        tags.append("in_place")
    else:
        tags.append("locomotion")
    if turn_rate > 1.0 and root_speed < float(th["static_root_speed_mps"]):
        tags.append("turning")

    # --- scene tags -----------------------------------------------------------
    if metrics.get("num_objects"):
        tags.append("has_object")
        if float(metrics.get("hand_object_contact_ratio", 0.0)) > 0.1:
            tags.append("interaction")
    if metrics.get("has_terrain"):
        tags.append("has_terrain")
        if float(metrics.get("terrain_height_range", 0.0)) > 0.2:
            tags.append("parkour")

    return tags


def assign_dataset_tags(clips: list[Any], cfg: dict[str, Any]) -> None:
    """Append distribution-relative tags to every clip (mutates ``clip.tags``).

    ``clips`` is a list of :class:`~hhtools.analysis.clip.AnalyzableClip`.
    """
    if not clips:
        return
    th = cfg["thresholds"]
    quants = th["dynamics_quantiles"]

    complexity = np.array([float(c.metrics.get("complexity", 0.0)) for c in clips])
    accel = np.array([float(c.metrics.get("joint_accel_energy", 0.0)) for c in clips])
    leg = np.array([float(c.metrics.get("leg_energy", 0.0)) for c in clips])

    finite = complexity[np.isfinite(complexity)]
    if finite.size == 0:
        return
    q_lo, q_hi, q_burst = (float(np.quantile(finite, x)) for x in quants)
    leg_q = float(np.quantile(leg[np.isfinite(leg)], float(th["leg_static_quantile"]))) \
        if np.isfinite(leg).any() else 0.0
    accel_hi = float(np.quantile(accel[np.isfinite(accel)], 0.75)) \
        if np.isfinite(accel).any() else 0.0

    for c in clips:
        cx = float(c.metrics.get("complexity", 0.0))
        if cx < q_lo:
            dyn = "static" if cx <= q_lo * 0.5 else "low_dynamic"
        elif cx < q_hi:
            dyn = "mid_dynamic"
        else:
            dyn = "high_dynamic"
        c.tags.append(dyn)

        # Burst: high acceleration energy but only moderate sustained complexity.
        if cx >= q_burst and float(c.metrics.get("joint_accel_energy", 0.0)) > accel_hi:
            c.tags.append("burst")

        if float(c.metrics.get("leg_energy", 0.0)) <= leg_q:
            c.tags.append("leg_static")

        # Dedupe while preserving order.
        seen: dict[str, None] = {}
        c.tags = [t for t in c.tags if not (t in seen or seen.setdefault(t, None))]


def all_known_tags() -> list[str]:
    """Stable ordering for UI chips."""
    return [
        "quality_ok",
        "quality_warn",
        "quality_bad",
        "static",
        "low_dynamic",
        "mid_dynamic",
        "high_dynamic",
        "burst",
        "in_place",
        "locomotion",
        "turning",
        "leg_static",
        "jump",
        "flip",
        "has_object",
        "interaction",
        "has_terrain",
        "parkour",
    ]


__all__ = ["all_known_tags", "assign_clip_tags", "assign_dataset_tags"]
