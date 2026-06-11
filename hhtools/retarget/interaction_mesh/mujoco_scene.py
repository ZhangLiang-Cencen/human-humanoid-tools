# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo scene handle for interaction-mesh retargeting.

Reuses the :class:`~hhtools.robot.loader.URDFRobotModel` compile path so mesh
search / URDF quirks stay identical to the Newton pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from hhtools.robot.loader import URDFRobotModel

__all__ = ["MujocoScene", "require_mujoco_model"]

_log = logging.getLogger(__name__)


def require_mujoco_model(robot: URDFRobotModel):
    """Return ``robot.mujoco_model``, retrying compilation if initially None.

    The initial ``load_robot`` compilation can fail silently (caught exception
    sets ``mujoco_model = None``) — e.g. when the code path that injects
    ``<compiler meshdir=…/>`` was added after the robot was first loaded.
    This retry gives the interaction-mesh backend a second chance using the
    current code.
    """
    mj = robot.mujoco_model
    if mj is not None:
        return mj

    from hhtools.robot.loader import _compile_mjcf, _resolve_mesh_search_paths

    last_exc: Exception | None = None
    urdf_path = robot.preset.urdf_path
    if urdf_path is not None and urdf_path.is_file():
        try:
            search_paths = _resolve_mesh_search_paths(robot.preset)
            mj, mjcf_xml = _compile_mjcf(urdf_path, search_paths)
            robot.mujoco_model = mj
            robot.mjcf_xml = mjcf_xml
            _log.info(
                "On-demand URDF→MJCF compilation succeeded for %s",
                robot.preset.name,
            )
            return mj
        except Exception as exc:
            last_exc = exc
            _log.warning("On-demand URDF→MJCF compilation failed: %s", exc)

    detail = f" Compilation error: {last_exc}" if last_exc else ""
    raise RuntimeError(
        f"URDF did not compile to a MuJoCo model (mujoco_model is None) "
        f"for robot {robot.preset.name!r} "
        f"(urdf={robot.preset.urdf_path}).{detail} "
        f"Install mujoco and yourdfpy (``pip install hhtools[robot]``) and "
        f"verify the preset URDF resolves mesh paths."
    ) from last_exc


def _ensure_freejoint(model, robot: URDFRobotModel):
    """If *model* has no FREE joint, recompile from its MJCF with one injected.

    Fixed-base URDFs compile to hinge-only models; the interaction mesh
    optimizer needs a floating base to control root translation/rotation.
    Returns the (possibly new) model.

    We wrap all worldbody children in a new ``floating_base`` body that
    carries the ``<freejoint/>``, avoiding the "more than 6 dofs" error
    that would occur if we added the freejoint directly to a body that
    already has its own joints.

    Implementation note — **never** call ``mj_saveLastXML`` here. That
    API saves whichever model was *most recently compiled in the
    process*, ignoring the passed ``model`` argument (which only carries
    spec-option overrides, not the model identity).  In a multi-clip
    session that means the second retarget can pick up a stale collision
    scene XML from the first run — including its ``<asset><hfield
    file="…/hhtools_terrain_hfield_*.bin"/>`` reference, which has been
    deleted.  Recompiling that XML then crashes with
    ``Error opening file '.../hhtools_terrain_hfield_*.bin'`` even
    though the current clip has no terrain at all.

    Instead we always start from ``robot.mjcf_xml`` — the clean
    URDF→MJCF string the loader cached at startup — so the freejoint
    transform is independent of any other models compiled later.
    """
    import mujoco
    import os
    import tempfile
    import xml.etree.ElementTree as ET
    from pathlib import Path

    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        # Already has a free joint — return the cached MJCF if available so
        # downstream callers still get a clean XML snapshot to work with.
        existing_xml = getattr(robot, "mjcf_xml", "") or ""
        return model, existing_xml

    urdf_dir = robot.preset.urdf_path.parent
    base_xml = getattr(robot, "mjcf_xml", "") or ""
    if not base_xml:
        # Loader couldn't precompile; fall back to a one-shot compile so we
        # still get a clean snapshot.  ``mj_saveLastXML`` is intentionally
        # avoided — see docstring.
        from hhtools.robot.loader import _compile_mjcf, _resolve_mesh_search_paths

        _, base_xml = _compile_mjcf(
            robot.preset.urdf_path, _resolve_mesh_search_paths(robot.preset)
        )
    tree = ET.ElementTree(ET.fromstring(base_xml))

    root = tree.getroot()

    def _has_free_joint(body: ET.Element) -> bool:
        """True if *body* already contains a MuJoCo free joint.

        ``mj_saveLastXML`` may serialise a free joint either as
        ``<freejoint name="..."/>`` or as ``<joint type="free" name="..."/>``.
        The latter was not detected before, causing us to insert a duplicate
        ``root_free`` joint on recompilation.
        """
        if body.find("freejoint") is not None:
            return True
        for joint in body.findall("joint"):
            if joint.get("type", "") == "free":
                return True
        return False

    def _unique_joint_name(base: str = "root_free") -> str:
        used: set[str] = set()
        for elem in root.iter():
            nm = elem.get("name")
            if nm:
                used.add(nm)
        if base not in used:
            return base
        i = 1
        while f"{base}_{i}" in used:
            i += 1
        return f"{base}_{i}"

    # mj_saveLastXML may emit a <compiler meshdir="..."/> that uses the
    # temp file's parent as base.  Since we write the modified MJCF into
    # the same directory, force meshdir to "." (= urdf_dir) so asset
    # paths resolve the same way as the original compilation.
    compiler = root.find("compiler")
    if compiler is not None:
        saved_meshdir = compiler.get("meshdir", "")
        if saved_meshdir and saved_meshdir != ".":
            resolved = (Path(saved_meshdir)
                        if Path(saved_meshdir).is_absolute()
                        else urdf_dir / saved_meshdir)
            if not resolved.is_dir():
                compiler.set("meshdir", str(urdf_dir))
                _log.debug(
                    "fixed meshdir in saved MJCF: %r → %s", saved_meshdir, urdf_dir,
                )

    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("no <worldbody> in MJCF")

    existing_fb = wb.find("body[@name='floating_base']")
    if existing_fb is not None:
        if _has_free_joint(existing_fb):
            _log.info(
                "floating_base already present in saved MJCF — recompiling as-is"
            )
        else:
            ET.SubElement(existing_fb, "freejoint", name=_unique_joint_name())
    else:
        wrapper = ET.SubElement(wb, "body", name="floating_base")
        ET.SubElement(wrapper, "freejoint", name=_unique_joint_name())
        ET.SubElement(wrapper, "inertial", pos="0 0 0", mass="1e-4",
                      diaginertia="1e-6 1e-6 1e-6")
        for child in list(wb):
            if child is not wrapper:
                wb.remove(child)
                wrapper.append(child)

    fd2, tmp_new = tempfile.mkstemp(suffix=".xml", prefix="hhtools_im_free_", dir=urdf_dir)
    os.close(fd2)
    try:
        tree.write(tmp_new, encoding="utf-8", xml_declaration=True)
        new_model = mujoco.MjModel.from_xml_path(tmp_new)
        # Snapshot the XML *before* unlinking so callers (the collision
        # scene builder) have a clean self-contained MJCF they can
        # reuse without reaching for mj_saveLastXML.
        with open(tmp_new, encoding="utf-8") as fp:
            new_xml = fp.read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to recompile MJCF after adding freejoint for "
            f"robot {robot.preset.name!r}: {exc}"
        ) from exc
    finally:
        try:
            os.unlink(tmp_new)
        except OSError:
            pass

    _log.info(
        "added floating base (freejoint) for interaction mesh: "
        "nq %d→%d, nv %d→%d",
        model.nq, new_model.nq, model.nv, new_model.nv,
    )
    return new_model, new_xml


@dataclass
class MujocoScene:
    """Lightweight MuJoCo robot scene (robot-only; props added later).

    ``mjcf_xml`` is the **freejoint-augmented** MJCF string the model was
    compiled from.  Carrying it on the scene lets the collision-scene
    builder rebuild a derived model (with a heightfield ``<hfield>``,
    ground plane, etc.) without having to ``mj_saveLastXML`` — that API
    is process-global and would otherwise return whichever model was
    last compiled in the entire session, leaking stale paths across
    independent retarget runs.
    """

    robot: URDFRobotModel
    model: object  # mujoco.MjModel — typed as object to avoid hard import in stubs
    data: object  # mujoco.MjData
    mjcf_xml: str = ""  # freejoint-augmented MJCF; consumed by collision builder

    @classmethod
    def from_robot(cls, robot: URDFRobotModel) -> MujocoScene:
        import mujoco

        base_model = require_mujoco_model(robot)
        model, xml = _ensure_freejoint(base_model, robot)
        data = mujoco.MjData(model)
        return cls(robot=robot, model=model, data=data, mjcf_xml=xml)

    def forward(self, qpos: np.ndarray) -> None:
        """Set ``data.qpos`` (full model vector) and run ``mj_forward``."""
        import mujoco

        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must be ({self.model.nq},); got {qpos.shape}")
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
