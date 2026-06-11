"""Stripping a redundant ``world``/floating-base anchor from vendor URDFs.

SolidWorks/ROS exports often ship ``<link name="world"/>`` plus a
``type="floating"`` joint to the real base.  hhtools adds its own floating base
via ``add_urdf(floating=True)``; the vendor anchor therefore stacks a *second*
6-DoF root and the retargeted robot floats.  These tests lock in detection +
stripping (and that clean URDFs are left alone).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from hhtools.robot.urdf_normalize import (
    detect_redundant_floating_base_root,
    ensure_urdf_meshes_resolvable,
    strip_redundant_floating_base_root,
    urdf_needs_persisted_fixes,
)

_WORLD_FLOATING_URDF = """<?xml version="1.0"?>
<robot name="bot">
  <mujoco><compiler meshdir="." discardvisual="false"/></mujoco>
  <link name="world"/>
  <joint name="floating_base_joint" type="floating">
    <parent link="world"/>
    <child link="base_link"/>
  </joint>
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
"""

_CLEAN_URDF = """<?xml version="1.0"?>
<robot name="bot">
  <mujoco><compiler meshdir="." discardvisual="false"/></mujoco>
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
"""

# A ``fixed`` anchor adds zero DoF → a single, correct floating base.  hhtools
# leaves these untouched so working robots (e.g. Kuavo ``dummy_link``, X2
# ``base_link`` placeholder) are never disturbed.
_WORLD_FIXED_URDF = _WORLD_FLOATING_URDF.replace(
    'type="floating"', 'type="fixed"'
).replace("floating_base_joint", "fixed_base_joint")


def _write(tmp_path: Path, text: str) -> Path:
    urdf = tmp_path / "bot.urdf"
    urdf.write_text(text, encoding="utf-8")
    return urdf


def test_detect_world_floating_anchor(tmp_path: Path) -> None:
    root = ET.fromstring(_WORLD_FLOATING_URDF)
    assert detect_redundant_floating_base_root(root) == (
        "world",
        "floating_base_joint",
        "base_link",
    )


def test_fixed_anchor_is_left_untouched(tmp_path: Path) -> None:
    # A fixed anchor is a correct single floating base; stripping it could break
    # ik_map slots that point at the dummy root, so it must NOT be detected.
    root = ET.fromstring(_WORLD_FIXED_URDF)
    assert detect_redundant_floating_base_root(root) is None


def test_clean_urdf_has_no_anchor() -> None:
    root = ET.fromstring(_CLEAN_URDF)
    assert detect_redundant_floating_base_root(root) is None


def test_strip_makes_base_link_the_single_root(tmp_path: Path) -> None:
    urdf = _write(tmp_path, _WORLD_FLOATING_URDF)
    out = strip_redundant_floating_base_root(urdf, output_path=urdf)
    root = ET.parse(out).getroot()

    link_names = [l.get("name") for l in root.findall("link")]
    joint_names = [j.get("name") for j in root.findall("joint")]
    assert "world" not in link_names
    assert "floating_base_joint" not in joint_names

    child_links = {
        j.find("child").get("link")
        for j in root.findall("joint")
        if j.find("child") is not None
    }
    roots = [n for n in link_names if n not in child_links]
    assert roots == ["base_link"]
    # Idempotent: a second pass finds nothing to strip.
    assert detect_redundant_floating_base_root(root) is None


def test_strip_to_separate_output_keeps_source(tmp_path: Path) -> None:
    urdf = _write(tmp_path, _WORLD_FLOATING_URDF)
    out = tmp_path / "stripped.urdf"
    strip_redundant_floating_base_root(urdf, output_path=out)
    # Source untouched, destination stripped.
    assert detect_redundant_floating_base_root(ET.parse(urdf).getroot()) is not None
    assert detect_redundant_floating_base_root(ET.parse(out).getroot()) is None


def test_urdf_needs_persisted_fixes_flags_anchor(tmp_path: Path) -> None:
    urdf = _write(tmp_path, _WORLD_FLOATING_URDF)
    assert urdf_needs_persisted_fixes(urdf) is True


def test_ensure_urdf_meshes_resolvable_strips_anchor(tmp_path: Path) -> None:
    (tmp_path / "meshes").mkdir()
    urdf = _write(tmp_path, _WORLD_FLOATING_URDF)
    out = ensure_urdf_meshes_resolvable(urdf)
    root = ET.parse(out).getroot()
    assert detect_redundant_floating_base_root(root) is None
    assert "world" not in [l.get("name") for l in root.findall("link")]
