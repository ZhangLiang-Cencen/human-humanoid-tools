"""BVH (Biovision Hierarchy) parser and writer.

This is a clean-room implementation authored for this project; no code is copied from any
non-compatible source. The parser supports the common BVH dialects emitted by MotionBuilder,
3ds Max, Blender, and the LAFAN1 / SOMA datasets.

Key features:

- Captures the per-bone ``OFFSET`` as local rest translation.
- Parses arbitrary ``CHANNELS`` orderings, including a 6-channel root (position + rotation) and
  3-channel children (rotation only).
- Converts every frame into global joint positions + xyzw quaternions in the internal Z-up
  coordinate system (BVH files are Y-up; we rotate with a single matrix at load time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.math import rotation as R
from hhtools.core.motion import Motion
from hhtools.io.loader_progress import ProgressCallback, report_progress

_UNIT_ALIASES = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "inch": 0.0254,
    "inches": 0.0254,
    "foot": 0.3048,
    "feet": 0.3048,
}


# ---------------------------------------------------------------------- data


@dataclass
class _BVHJoint:
    name: str
    parent: int  # index into the joint list; -1 for root
    offset: NDArray  # (3,) float32 in source BVH units (not yet scaled)
    channels: list[str] = field(default_factory=list)
    is_end_site: bool = False


# ---------------------------------------------------------------------- parser


def load_bvh(
    path: str | Path,
    *,
    unit: str = "cm",
    target_up_axis: str = "Z",
    progress_callback: ProgressCallback | None = None,
) -> Motion:
    """Load a BVH file as a :class:`Motion`.

    Args:
        path: Path to the ``.bvh`` file.
        unit: Length unit of the source file. BVH offsets are usually in centimetres; we scale
            them to metres here. Accepts ``"m" | "cm" | "mm" | "inch" | ...``.
        target_up_axis: Internal up axis. Defaults to ``"Z"``, which matches the hhtools
            convention; pass ``"Y"`` to keep the BVH original orientation.
        progress_callback: Optional ``fn(frac, message)`` for long parses (web UI / cache).
    """
    path = Path(path)
    report_progress(progress_callback, 0.0, f"读取 BVH {path.name}…")
    text = path.read_text(encoding="utf-8", errors="ignore")
    scale = _UNIT_ALIASES.get(unit.lower())
    if scale is None:
        raise ValueError(f"Unknown BVH unit {unit!r}; known: {sorted(_UNIT_ALIASES)}")

    tokens = _tokenize(text)
    idx = 0
    if tokens[idx] != "HIERARCHY":
        raise ValueError(f"{path}: expected 'HIERARCHY' at the beginning of BVH")
    idx += 1

    joints: list[_BVHJoint] = []
    channel_count = 0
    idx = _parse_joint(tokens, idx, joints, parent=-1)

    for j in joints:
        if not j.is_end_site:
            channel_count += len(j.channels)

    if tokens[idx] != "MOTION":
        raise ValueError(f"{path}: expected 'MOTION' after hierarchy, got {tokens[idx]!r}")
    idx += 1

    if tokens[idx].lower() != "frames:":
        raise ValueError(f"{path}: expected 'Frames:' token, got {tokens[idx]!r}")
    idx += 1
    num_frames = int(tokens[idx])
    idx += 1

    if tokens[idx].lower() != "frame" or tokens[idx + 1].lower() != "time:":
        raise ValueError(f"{path}: expected 'Frame Time:' token")
    idx += 2
    frame_time = float(tokens[idx])
    idx += 1

    framerate = 1.0 / frame_time if frame_time > 0 else 30.0
    num_active_joints = sum(1 for j in joints if not j.is_end_site)
    report_progress(
        progress_callback, 0.25, f"解析 BVH 骨架 ({num_active_joints} 关节)…",
    )

    data_values = np.array([float(t) for t in tokens[idx:]], dtype=np.float32)
    expected = num_frames * channel_count
    if data_values.size != expected:
        raise ValueError(
            f"{path}: expected {expected} motion values ({num_frames} frames x "
            f"{channel_count} channels), got {data_values.size}"
        )
    data_values = data_values.reshape(num_frames, channel_count)
    report_progress(
        progress_callback, 0.55, f"处理 BVH 动作数据 ({num_frames} 帧)…",
    )

    # Scale offsets to metres.
    offsets = np.stack([j.offset for j in joints]).astype(np.float32) * scale

    # Build the hierarchy (excluding End Sites).
    active_mask = [not j.is_end_site for j in joints]
    active_joints = [j for j in joints if not j.is_end_site]
    active_indices = [i for i, a in enumerate(active_mask) if a]
    old_to_new = {old: new for new, old in enumerate(active_indices)}

    active_names = [j.name for j in active_joints]
    parent_indices = np.array(
        [-1 if j.parent == -1 else old_to_new[j.parent] for j in active_joints],
        dtype=np.int32,
    )
    hierarchy = Hierarchy.from_parent_indices(active_names, parent_indices)

    # Extract per-joint, per-frame local position + quaternion from the channel stream.
    # The root typically has 6 channels (3 position + 3 rotation), children 3 (rotation).
    local_positions = np.zeros((num_frames, len(active_joints), 3), dtype=np.float32)
    local_quats = np.zeros((num_frames, len(active_joints), 4), dtype=np.float32)
    local_quats[..., 3] = 1.0
    # The default local translation for non-root joints is the rest offset.
    for new_i, old_i in enumerate(active_indices):
        local_positions[:, new_i] = offsets[old_i]

    cursor = 0
    for old_i, joint in enumerate(joints):
        if joint.is_end_site:
            continue
        new_i = old_to_new[old_i]
        if not joint.channels:
            continue
        ch_block = data_values[:, cursor : cursor + len(joint.channels)]
        cursor += len(joint.channels)

        # Gather rotation channels (preserve BVH order for the intrinsic composition).
        rot_order = ""
        rot_values = []
        trans_xyz = [None, None, None]
        for ch_i, ch in enumerate(joint.channels):
            lc = ch.lower()
            if lc == "xposition":
                trans_xyz[0] = ch_block[:, ch_i]
            elif lc == "yposition":
                trans_xyz[1] = ch_block[:, ch_i]
            elif lc == "zposition":
                trans_xyz[2] = ch_block[:, ch_i]
            elif lc == "xrotation":
                rot_order += "X"
                rot_values.append(ch_block[:, ch_i])
            elif lc == "yrotation":
                rot_order += "Y"
                rot_values.append(ch_block[:, ch_i])
            elif lc == "zrotation":
                rot_order += "Z"
                rot_values.append(ch_block[:, ch_i])
            else:
                raise ValueError(f"{path}: unknown channel {ch!r}")

        if any(v is not None for v in trans_xyz):
            zero = np.zeros(num_frames, dtype=np.float32)
            tx = trans_xyz[0] if trans_xyz[0] is not None else zero
            ty = trans_xyz[1] if trans_xyz[1] is not None else zero
            tz = trans_xyz[2] if trans_xyz[2] is not None else zero
            # Root translation is in source units -> metres
            local_positions[:, new_i] = np.stack([tx, ty, tz], axis=-1) * scale

        if rot_order:
            angles = np.stack(rot_values, axis=-1)  # (num_frames, 3)
            local_quats[:, new_i] = R.bvh_euler_to_quat(angles, rot_order, degrees=True)

    # Forward kinematics to get global positions + quaternions.
    positions, quats = _forward_kinematics(
        hierarchy=hierarchy, local_positions=local_positions, local_quats=local_quats
    )
    report_progress(progress_callback, 0.9, "BVH 正向运动学…")

    # Rotate from BVH Y-up to target up-axis.
    if target_up_axis.upper() != "Y":
        rot_mat = R.up_axis_rotation("Y", target_up_axis.upper())
        positions = positions @ rot_mat.T
        # Apply the same rotation to global quaternions by pre-multiplying.
        rot_quat = Q.from_matrix(rot_mat)
        quats = Q.multiply(np.broadcast_to(rot_quat, quats.shape), quats)

    bvh_rotation_orders: dict[str, str] = {}
    for j in joints:
        if j.is_end_site:
            continue
        rot_order = _rotation_order_from_channels(j.channels)
        if rot_order:
            bvh_rotation_orders[j.name] = rot_order

    report_progress(progress_callback, 1.0, f"BVH 加载完成 ({path.name})")
    return Motion(
        name=path.stem,
        hierarchy=hierarchy,
        positions=positions.astype(np.float32),
        quaternions=quats.astype(np.float32),
        framerate=framerate,
        up_axis=target_up_axis.upper(),  # type: ignore[arg-type]
        source_format="bvh",
        meta={
            "source_path": str(path),
            "source_unit": unit,
            "source_up_axis": "Y",
            "frame_time": frame_time,
            "num_channels": channel_count,
            "bvh_rotation_orders": bvh_rotation_orders,
        },
    )


# ---------------------------------------------------------------------- writer


def save_bvh(motion: Motion, path: str | Path, *, degrees: bool = True) -> None:
    """Write a :class:`Motion` to a minimal BVH file.

    Re-projects global transforms into parent-local frames.  When
    ``motion.meta[\"bvh_rotation_orders\"]`` is present (populated by
    :func:`load_bvh`), the original per-joint rotation channel order is
    preserved so round-tripped clips stay quaternion-compatible with their
    source dialect (e.g. MOCAP ``YXZ`` vs LAFAN ``ZYX``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rot_orders: dict[str, str] = dict(
        motion.meta.get("bvh_rotation_orders") or {}
    )

    # Compute local transforms via inverse FK at rest.
    local_positions, local_quats = _inverse_forward_kinematics(
        motion.hierarchy, motion.positions, motion.quaternions
    )

    lines: list[str] = ["HIERARCHY"]
    roots = motion.hierarchy.root_indices()

    def _channels_line(is_root: bool, bone_name: str) -> str:
        order = rot_orders.get(bone_name, "ZYX")
        rot_names = [f"{axis}rotation" for axis in order]
        if is_root:
            return "CHANNELS 6 Xposition Yposition Zposition " + " ".join(rot_names)
        return "CHANNELS 3 " + " ".join(rot_names)

    def write_joint(idx: int, depth: int, is_root: bool) -> None:
        pad = "\t" * depth
        label = "ROOT" if is_root else "JOINT"
        bone_name = motion.hierarchy.bone_names[idx]
        lines.append(f"{pad}{label} {bone_name}")
        lines.append(f"{pad}{{")
        offs = local_positions[0, idx]
        lines.append(f"{pad}\tOFFSET {offs[0]:.6f} {offs[1]:.6f} {offs[2]:.6f}")
        lines.append(f"{pad}\t{_channels_line(is_root, bone_name)}")
        children = motion.hierarchy.children(idx)
        if children:
            for c in children:
                write_joint(c, depth + 1, is_root=False)
        else:
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    for r in roots:
        write_joint(r, 0, is_root=True)

    lines.append("MOTION")
    lines.append(f"Frames: {motion.num_frames}")
    lines.append(f"Frame Time: {1.0 / motion.framerate:.8f}")

    # Encode each frame: root (pos + per-joint euler order) + children.
    for f in range(motion.num_frames):
        row: list[str] = []
        for idx in _dfs_order(motion.hierarchy):
            lp = local_positions[f, idx]
            lq = local_quats[f, idx]
            bone_name = motion.hierarchy.bone_names[idx]
            order = rot_orders.get(bone_name, "ZYX")
            euler = np.asarray(
                R.quat_to_bvh_euler_for_write(lq, order, degrees=degrees),
                dtype=np.float64,
            ).reshape(3)
            if motion.hierarchy.is_root(idx):
                row.extend([f"{lp[0]:.6f}", f"{lp[1]:.6f}", f"{lp[2]:.6f}"])
            row.extend([f"{euler[0]:.6f}", f"{euler[1]:.6f}", f"{euler[2]:.6f}"])
        lines.append(" ".join(row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- helpers

_TOKEN_RE = re.compile(r"\{|\}|[^\s{}]+")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        for m in _TOKEN_RE.finditer(line):
            tokens.append(m.group(0))
    return tokens


def _parse_joint(tokens: list[str], idx: int, joints: list[_BVHJoint], parent: int) -> int:
    tag = tokens[idx]
    if tag == "ROOT":
        idx += 1
        name = tokens[idx]
        idx += 1
    elif tag == "JOINT":
        idx += 1
        name = tokens[idx]
        idx += 1
    elif tag == "End":
        idx += 1
        assert tokens[idx].lower() == "site", f"Expected 'Site', got {tokens[idx]!r}"
        idx += 1
        name = f"{joints[parent].name}_end"
    else:
        raise ValueError(f"Unexpected BVH tag {tag!r} at token index {idx}")

    if tokens[idx] != "{":
        raise ValueError(f"Expected '{{' after joint header, got {tokens[idx]!r}")
    idx += 1

    is_end_site = tag == "End"
    joint = _BVHJoint(
        name=name,
        parent=parent,
        offset=np.zeros(3, dtype=np.float32),
        channels=[],
        is_end_site=is_end_site,
    )
    joints.append(joint)
    my_index = len(joints) - 1

    while idx < len(tokens) and tokens[idx] != "}":
        tok = tokens[idx]
        if tok.upper() == "OFFSET":
            idx += 1
            joint.offset = np.array(
                [float(tokens[idx]), float(tokens[idx + 1]), float(tokens[idx + 2])],
                dtype=np.float32,
            )
            idx += 3
            # Some BVH dialects (notably the ones emitted by blender /
            # upstream soma-retargeter's ``*_zero_frame0.bvh``) tack
            # additional floats onto the OFFSET line to carry per-joint
            # bind rotations.  Standard 3.0 BVH only expects 3 floats,
            # so consume any trailing numeric tokens until we hit the
            # next keyword; dropping the rotation extras is safe because
            # the same rotations appear again in the MOTION section.
            while idx < len(tokens):
                try:
                    float(tokens[idx])
                except ValueError:
                    break
                idx += 1
        elif tok.upper() == "CHANNELS":
            idx += 1
            n = int(tokens[idx])
            idx += 1
            joint.channels = [tokens[idx + k] for k in range(n)]
            idx += n
        elif tok in ("JOINT", "End", "ROOT"):
            idx = _parse_joint(tokens, idx, joints, parent=my_index)
        else:
            raise ValueError(f"Unknown BVH token {tok!r}")

    if tokens[idx] != "}":
        raise ValueError(f"Expected '}}' at end of joint, got {tokens[idx]!r}")
    idx += 1
    return idx


def _forward_kinematics(
    hierarchy: Hierarchy, local_positions: NDArray, local_quats: NDArray
) -> tuple[NDArray, NDArray]:
    """Batched forward kinematics across time.

    Returns ``(global_positions, global_quats)`` with shapes ``(F, N, 3)`` and ``(F, N, 4)``.
    """
    num_frames, num_bones, _ = local_positions.shape
    gp = np.empty_like(local_positions)
    gq = np.empty_like(local_quats)
    for i in range(num_bones):
        p = hierarchy.parent(i)
        if p == -1:
            gq[:, i] = local_quats[:, i]
            gp[:, i] = local_positions[:, i]
        else:
            gq[:, i] = Q.multiply(gq[:, p], local_quats[:, i])
            gp[:, i] = gp[:, p] + Q.rotate(gq[:, p], local_positions[:, i])
    return gp, gq


def _inverse_forward_kinematics(
    hierarchy: Hierarchy, global_positions: NDArray, global_quats: NDArray
) -> tuple[NDArray, NDArray]:
    """Convert global positions + quaternions back to parent-local form."""
    num_frames, num_bones, _ = global_positions.shape
    lp = np.empty_like(global_positions)
    lq = np.empty_like(global_quats)
    for i in range(num_bones):
        p = hierarchy.parent(i)
        if p == -1:
            lq[:, i] = global_quats[:, i]
            lp[:, i] = global_positions[:, i]
        else:
            parent_q_inv = Q.conjugate(global_quats[:, p])
            lq[:, i] = Q.multiply(parent_q_inv, global_quats[:, i])
            delta = global_positions[:, i] - global_positions[:, p]
            lp[:, i] = Q.rotate(parent_q_inv, delta)
    return lp, lq


def _dfs_order(hierarchy: Hierarchy) -> list[int]:
    order: list[int] = []
    stack = list(reversed(hierarchy.root_indices()))
    while stack:
        i = stack.pop()
        order.append(i)
        children = hierarchy.children(i)
        stack.extend(reversed(children))
    return order


def _rotation_order_from_channels(channels: list[str]) -> str:
    """Extract intrinsic rotation order from a BVH ``CHANNELS`` list."""

    order = ""
    for ch in channels:
        lc = ch.lower()
        if lc.endswith("rotation"):
            order += lc[0].upper()
    return order


def _quat_to_euler_zyx(q: NDArray, *, degrees: bool) -> NDArray:
    """Convert an xyzw quaternion to intrinsic ZYX Euler angles (z, y, x order)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Intrinsic Z-Y-X Euler = extrinsic X-Y-Z on column vector; derive with standard formulas.
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
    # Order is (Zrot, Yrot, Xrot) to match the CHANNELS declaration.
    return np.stack([yaw, pitch, roll], axis=-1)
