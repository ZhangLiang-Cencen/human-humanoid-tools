"""MuJoCo compiler block injection for vendor URDFs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from hhtools.robot.urdf_normalize import ensure_mujoco_compiler_block


def test_ensure_mujoco_compiler_block_inserts_mujoco(tmp_path: Path) -> None:
    urdf = tmp_path / "bot.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="bot">
  <link name="base"/>
</robot>
""",
        encoding="utf-8",
    )
    (tmp_path / "meshes").mkdir()
    out = ensure_mujoco_compiler_block(urdf)
    root = ET.parse(out).getroot()
    mujoco = next(child for child in root if child.tag == "mujoco")
    compiler = mujoco.find("compiler")
    assert compiler is not None
    assert compiler.get("meshdir") == "meshes"
    assert compiler.get("discardvisual") == "false"
