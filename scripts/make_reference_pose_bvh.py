"""Author a clean, symmetric T-pose ``*_zero_frame0.bvh`` for a source rig.

The retarget pipeline calibrates and scales every source format against a
*rest pose* — the answer to "what does this skeleton look like at rest?".
SOMA / Xsens ship a bundled ``assets/reference_poses/*_zero_frame0.bvh`` for
this.  LAFAN / 20260429-mocap historically synthesised their rest on the fly
from each clip's median bone lengths.  This script bakes that same synthesised
T-pose into a bundled BVH so those formats use the identical file-based
workflow (and so the calibration overlay is fully clip-independent).

The synthesised T-pose is deliberately left-right **symmetric** and upright,
which avoids the right-shoulder asymmetry artefact that captured zero frames
bake in.

Usage::

    python scripts/make_reference_pose_bvh.py \
        --clip assets/motions/mimic/LAFAN/dance1_subject2.bvh \
        --out assets/reference_poses/lafan_zero_frame0.bvh
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from hhtools.core.coord import to_up_axis
from hhtools.core.math import quaternion as Q
from hhtools.io import load_motion
from hhtools.io.bvh import load_bvh, save_bvh


def _authoring_rest_pose(clip):  # type: ignore[no-untyped-def]
    """Build the Z-up rest snapshot written into a bundled ``*_zero_frame0.bvh``."""

    import numpy as np

    from hhtools.retarget.newton_basic.human_aliases import (
        is_mixamo_cmu_like,
        is_mocap_spine3_bvh_like,
    )
    from hhtools.retarget.newton_basic.rest_pose import (
        _synthesise_tpose_from_bone_lengths,
        rest_pose_from_motion_bind,
    )

    bind = rest_pose_from_motion_bind(clip, source_tag="authoring_tpose")
    if is_mocap_spine3_bvh_like(clip.hierarchy.bone_names):
        # MOCAP clips start standing; pelvis / leg bind twist differs ~180° from
        # a synthesized T-pose even when foot positions match.  Copy standing
        # leg / pelvis quaternions from frame 0; keep T-pose arms for calibration.
        leg_bones = {
            "Hips",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        }
        quat = np.asarray(bind.quaternions, dtype=np.float32).copy()
        for name in leg_bones:
            if name in bind.bone_names and name in clip.hierarchy.bone_names:
                ci = clip.hierarchy.bone_names.index(name)
                ri = bind.bone_names.index(name)
                quat[ri] = clip.quaternions[0, ci]
        return replace(bind, quaternions=quat)

    if is_mixamo_cmu_like(clip.hierarchy.bone_names):
        # Bind FK for LAFAN / Mixamo puts the leg chain along +Z (feet above
        # hips), which breaks the calibration overlay (feet-on-floor layout).
        # Keep bind quaternions (T-pose arms) but lay bones out with the
        # bone-length synthesiser (−Z legs, ±X arms).
        parent_idx = np.asarray(clip.hierarchy.parent_indices, dtype=np.int64)
        synth_pos = _synthesise_tpose_from_bone_lengths(clip, parent_idx)
        return replace(bind, positions=synth_pos)

    return bind


def _build_tpose_motion(clip):  # type: ignore[no-untyped-def]
    """Single-frame Z-up :class:`Motion` holding the synthesised T-pose.

    The synthesised rest is upright and left-right **symmetric** (positions laid
    out from the subject's median bone lengths, quaternions pointing each bone
    along its T-pose direction).  ``save_bvh`` re-projects these world transforms
    into parent-local frames using the *same* world quaternions, so forward
    kinematics on import reconstructs them exactly — the symmetric pose is
    preserved verbatim (no per-rig FK refit, which would re-introduce the
    skeleton's own small left/right offset asymmetries).
    """
    rest = _authoring_rest_pose(clip)
    pos = np.asarray(rest.positions, dtype=np.float32)[None]
    quat = Q.normalize(np.asarray(rest.quaternions, dtype=np.float32))[None]
    return replace(
        clip,
        positions=pos,
        quaternions=quat,
        up_axis="Z",
        meta={**clip.meta, "reference_pose": "synthesised_tpose"},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="Representative clip of the rig.")
    ap.add_argument("--out", required=True, help="Destination *_zero_frame0.bvh.")
    args = ap.parse_args()

    clip = load_motion(args.clip)
    tpose_z = _build_tpose_motion(clip)

    # Preserve the source rig's BVH rotation channel order (e.g. MOCAP YXZ).
    rot_orders = dict(clip.meta.get("bvh_rotation_orders") or {})
    tpose_z = replace(
        tpose_z,
        meta={**tpose_z.meta, "bvh_rotation_orders": rot_orders},
    )

    # ``save_bvh`` writes positions verbatim while ``load_bvh`` (a) always treats
    # a BVH as Y-up and rotates Y->Z, and (b) scales offsets cm->m (``unit="cm"``)
    # on import.  Emit Y-up **in centimetres** so the round-trip lands back on
    # the original Z-up, metre-scale T-pose.
    tpose_y = to_up_axis(tpose_z, "Y")
    tpose_y_cm = replace(tpose_y, positions=tpose_y.positions * 100.0)
    out = Path(args.out)
    save_bvh(tpose_y_cm, out)

    # Verify the round-trip reproduces the authored T-pose.
    back = load_bvh(out)
    n = min(back.num_bones, tpose_z.num_bones)
    dp = float(np.abs(back.positions[0, :n] - tpose_z.positions[0, :n]).max())
    dq = float(np.abs(back.quaternions[0, :n] - tpose_z.quaternions[0, :n]).max())
    dq = min(dq, float(np.abs(back.quaternions[0, :n] + tpose_z.quaternions[0, :n]).max()))
    print(
        f"wrote {out}  bones={tpose_z.num_bones}  "
        f"roundtrip_max_pos_err={dp:.4f} m  quat_err={dq:.4f}"
    )
    if dp > 1e-2 or dq > 1e-2:
        raise SystemExit(
            f"round-trip error too large: pos={dp:.4f} m quat={dq:.4f}"
        )


if __name__ == "__main__":
    main()
