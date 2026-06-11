"""Core abstractions for the robot side of hhtools.

This module is deliberately minimal — it defines *what* a robot is to the rest of
hhtools (dataclasses + an ABC), and leaves *how* we load / render / IK each one
to :mod:`hhtools.robot.loader`, :mod:`hhtools.robot.registry`, and the viewer.

Design notes:

* **URDF is the authoritative input.**  Every registered robot has a URDF on
  disk; MJCF is a derived artefact we compile at load time (see
  :func:`hhtools.robot.loader.load_robot` and the `mjcf_xml` attribute on
  :class:`URDFRobotModel`).  Users only ever edit URDF + ``robot.yaml``.
* **``robot.yaml`` is the *only* file a user touches to add a new robot.**
  :class:`RobotPreset` is the deserialised form.  The ABC never peeks inside
  it — it's opaque payload the higher layers (registry + mapping editor) can
  read without needing the URDF to be loaded.
* **No IK / no motion in this file.**  ``retarget_newton_basic`` will add
  :class:`RetargetPipeline` objects that consume a :class:`RobotModel`, but
  the model itself is pure-description: links, joints, DOF schema, ik_map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# --------------------------------------------------------------------------- link / joint


@dataclass(frozen=True)
class LinkInfo:
    """A rigid body in the URDF.

    We keep this minimal: geometry is owned by the loader (yourdfpy gives us a
    ``trimesh.Scene``) and FK by Newton/MuJoCo.  This dataclass is only what we
    need for *naming, topology, and UI enumeration*.
    """

    name: str
    parent: str | None          # None for the root link
    child_joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointInfo:
    """A single joint in the URDF (one DOF for revolute/prismatic, 0 for fixed).

    ``axis`` is the joint rotation axis in the parent frame (unit vector for
    revolute; for fixed joints it's ``(0,0,0)``).  ``limit_lower`` / ``upper``
    are in radians (revolute) or meters (prismatic); ``None`` for unlimited or
    fixed joints.  This matches URDF spec.
    """

    name: str
    joint_type: Literal["revolute", "continuous", "prismatic", "fixed", "floating", "planar"]
    parent_link: str
    child_link: str
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_lower: float | None = None
    limit_upper: float | None = None
    velocity_limit: float | None = None
    effort_limit: float | None = None

    @property
    def is_actuated(self) -> bool:
        """Whether this joint contributes a scalar DOF to CSV / MuJoCo IK.

        Floating/planar roots are handled separately (7-DoF ``qpos`` block);
        they must not appear in ``dof_order`` or ``robot_dof_names``.
        """
        return self.joint_type not in ("fixed", "floating", "planar")


# --------------------------------------------------------------------------- preset payload


@dataclass
class RobotPreset:
    """Deserialised ``configs/robots/<name>/robot.yaml``.

    Kept as a plain dataclass (not pydantic) so the rest of the codebase can
    import it without pulling pydantic into the startup path.  The registry
    does coarse validation (required keys, path existence) when scanning.

    Attributes:
        name: Stable identifier used on the CLI and in log lines.  Must match
            the enclosing directory name — the registry asserts this so two
            robots can't collide on ``name``.
        display_name: Pretty label for UI dropdowns; falls back to ``name``.
        root_dir: Absolute path to the ``configs/robots/<name>/`` folder.
            All relative paths in the yaml resolve against this.
        urdf_path: Absolute path to the URDF.  ``None`` when the yaml points
            at a file that doesn't exist yet — we deliberately don't fail
            registration for this so users see placeholder presets in the UI
            with a clear "URDF missing" state (e.g. ``unitree_g1`` before the
            user drops a URDF in).
        mesh_search_paths: Absolute directories added to the yourdfpy search
            path during URDF parsing.  Lets URDF ``<mesh filename="...">``
            tags use relative paths without assuming where the file will live
            at install time.
        ik_map: ``canonical_human_joint_name -> robot_link_name``.  Opaque to
            this layer — consumed by the retargeter.
        weights: ``{"t_weight": {...}, "r_weight": {...}}`` per-joint
            translational / rotational IK weights.  Opaque here.
        rest_offsets: Axis-angle offsets applied to the retargeted pose before
            IK — built by the joint mapping editor.  Opaque.
        feet: Foot contact link names + ground height tuning for the feet
            stabilizer.  Opaque.
        length_scale: Multiplicative factor to apply to URDF-provided lengths
            at load time — useful for URDFs authored in millimetres.
        up_axis / forward_axis: Orientation hints; the viewer uses these to
            place the initial camera, and the retargeter uses them to align
            the human skeleton with the robot body on first load.
        dof_order: Explicit ordering of actuated joints for CSV export.  If
            empty the loader will derive one from URDF parsing order, but
            IK + retarget pipelines *require* this to be explicit (so that a
            CSV that was trained on one order doesn't silently break on a
            later URDF rebuild).
        meta: Anything else the yaml contains, preserved for forward compat.
    """

    name: str
    display_name: str
    root_dir: Path
    urdf_path: Path | None
    mesh_search_paths: tuple[Path, ...] = ()
    ik_map: dict[str, str] = field(default_factory=dict)
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    rest_offsets: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Per-link smoothing weights fed to
    # :class:`~hhtools.retarget.newton_basic.ik_objectives.IKSmoothJointFilter`.
    # Keys are robot link names (e.g. ``left_shoulder_roll_link``); values
    # in ``[0, 1]`` multiply the joint-filter residual for the coord(s)
    # driven by that link's joint.  Typical usage: push the redundant
    # yaw / outer-gimbal DOFs on shoulders & hips toward the middle of
    # their joint limit so IK picks the "natural" solution in the null
    # space of the position-only roll-link targets.  Empty == no extra
    # regularisation (only :class:`IKObjectiveJointLimit` hinges act).
    smooth_joint_filter_masks: dict[str, float] = field(default_factory=dict)
    feet: dict[str, Any] = field(default_factory=dict)
    length_scale: float = 1.0
    up_axis: Literal["X", "Y", "Z"] = "Z"
    forward_axis: Literal["X", "Y", "Z"] = "X"
    dof_order: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_urdf(self) -> bool:
        """Convenience: whether a concrete URDF is on disk.

        ``False`` when the preset exists but the URDF file is missing — the UI
        still shows the preset greyed-out so users know *which* presets are
        awaiting a user-supplied URDF (see ``configs/robots/unitree_g1`` as
        the canonical example).
        """
        return self.urdf_path is not None and self.urdf_path.is_file()


# --------------------------------------------------------------------------- ABC


class RobotModel(ABC):
    """Abstract interface the retargeter + viewer talk to.

    The ABC is intentionally tiny: everything non-trivial (mesh geometry,
    Newton compile, MuJoCo compile) lives behind ``urdf_scene`` / ``mjcf_xml``
    on the concrete :class:`URDFRobotModel` so callers can depend on the
    abstract type when they only need topology/DOF info — pure analytics
    paths (``hhtools/analytics/robot_metrics.py``) don't need yourdfpy or
    mujoco imported just to enumerate DOF names.
    """

    @property
    @abstractmethod
    def preset(self) -> RobotPreset: ...

    @property
    @abstractmethod
    def links(self) -> tuple[LinkInfo, ...]: ...

    @property
    @abstractmethod
    def joints(self) -> tuple[JointInfo, ...]: ...

    @property
    @abstractmethod
    def base_link(self) -> str:
        """Name of the root link (whatever URDF parents everything to)."""

    @property
    @abstractmethod
    def actuated_joints(self) -> tuple[JointInfo, ...]:
        """Subset of ``joints`` that contributes a DOF to the CSV.

        Order matches ``preset.dof_order`` when it's set; otherwise falls
        back to URDF parse order.
        """

    # ---- convenience enumerations (derived — not abstract) --------------------

    def link_names(self) -> tuple[str, ...]:
        return tuple(link.name for link in self.links)

    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    def dof_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.actuated_joints)

    def __repr__(self) -> str:  # pragma: no cover — trivial
        return (
            f"<{type(self).__name__} name={self.preset.name!r} "
            f"links={len(self.links)} dof={len(self.actuated_joints)}>"
        )
