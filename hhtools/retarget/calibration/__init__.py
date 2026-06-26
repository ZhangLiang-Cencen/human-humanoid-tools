"""Robot↔human retargeting calibration.

Establishes a one-time, per-robot alignment between:

* a known, reference-frame human pose (e.g. ``smpl`` T-pose,
  ``soma_bvh`` arms-down rest, or ``lafan_bvh`` near-T-pose), and
* the same robot posed via manually dialled actuated-joint angles with
  the floating base at identity.

When the user confirms the two poses match visually, a yaml file is
written as ``retarget_calibration_<reference>.yaml`` next to the URDF
(one calibration per robot **and** per reference format).  At
retarget time, per-canonical-joint scales + orientation offsets are
re-derived in closed form from the stored joint-angle configuration so
that the source motion's frame 0 lines up exactly with the robot's
calibrated pose.  Subsequent frames flow through the scaler as
"relative rotation from motion-frame-0 composed with the calibrated
orientation offset".

Compared to the ad-hoc first-frame heuristic this replaces, calibration
is explicit, inspectable (``git diff`` the YAML), and — once done per
robot — amortises across every source motion retargeted to that robot.
"""

from __future__ import annotations

from hhtools.retarget.calibration.calibration import (
    RobotRetargetCalibration,
    build_scaler_config_from_calibration,
    build_scaler_config_soma_style,
    calibration_path_for,
    derive_calibration_params,
    effective_retarget_human_height,
    load_calibration,
    normalize_calibration_reference,
    repair_apose_calibration_for_straight_t_reference,
    resolve_calibration_file,
    save_calibration,
)
from hhtools.retarget.calibration.reference import (
    HumanReferencePose,
    ReferenceName,
    build_motion_reference,
    load_reference_pose,
    list_reference_names,
    reference_pose_from_motion_frame0_quantized,
)

__all__ = [
    "HumanReferencePose",
    "ReferenceName",
    "RobotRetargetCalibration",
    "build_motion_reference",
    "build_scaler_config_from_calibration",
    "build_scaler_config_soma_style",
    "calibration_path_for",
    "derive_calibration_params",
    "effective_retarget_human_height",
    "list_reference_names",
    "load_calibration",
    "load_reference_pose",
    "reference_pose_from_motion_frame0_quantized",
    "normalize_calibration_reference",
    "repair_apose_calibration_for_straight_t_reference",
    "resolve_calibration_file",
    "save_calibration",
]
