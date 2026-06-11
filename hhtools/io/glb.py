"""GLB / glTF 2.0 skeletal-animation importer.

This rewrite is inspired by (but not copied from) Meta's `ai4animationpy.Import.GLBImporter`:
we keep their two-stage architecture (parse every ``nodes[*]`` entry, run FK over the full
node graph, then index out ``skins[0].joints`` from the result) because on real production
files (e.g. the Cranberry dance rig shipped under ``assets/motions/mimic/GLB/``) any
non-joint control / helper node may carry a rest translation or an animated scale that
propagates down to the skinned joints — collapsing the tree to joints-only silently drops
those contributions.

Key differences from the reference:

* We sample animation channels strictly per the glTF 2.0 spec:

  * ``LINEAR`` rotation tracks use slerp (shortest-arc) rather than per-component ``lerp``.
  * ``STEP`` is left-constant across an interval.
  * ``CUBICSPLINE`` is degraded to ``LINEAR`` on the middle "value" row; the affected
    ``joint.path`` tuples are exposed via :attr:`Motion.meta["cubicspline_fallback"]` so
    downstream pipelines can decide whether to re-encode.

* Node scale is propagated by accumulating the *parent* chain's per-frame scale and
  baking that factor into the child's local translation before quaternion FK.  This
  keeps the output ``positions`` dimensionally correct (Cranberry's ``body_world`` root
  joint has animated scale ≈ 0.917 — every descendant position must be ~0.917× the raw
  value, or the character is too tall by ~9%).

* We support JSON ``.gltf`` with external ``.bin`` siblings, ``data:`` URIs, and
  embedded GLB binary chunks transparently via ``pygltflib.GLTF2.get_data_from_buffer_uri``.

Deliberate non-goals for this step:

* Skinned mesh vertex buffers, materials, morph targets.  Those are tracked under the
  ``human_viz`` TODO — skeleton data is what ``hhtools`` retarget / analytics pipelines
  actually need today.
* Non-uniform scale that commutes with subsequent rotations (strictly speaking a
  full-affine matrix FK is required for fidelity here).  We bake non-uniform scales
  into child translations anyway — a warning is written to
  ``meta["scale_nonuniform"]`` when this approximation is activated.

Registered against ``.glb`` and ``.gltf`` on import so ``load_motion(p)`` dispatches here
automatically, matching the rest of the ``hhtools.io`` surface.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.math import rotation as R
from hhtools.core.motion import Motion
from hhtools.core.skinning import SkinnedMesh

# glTF accessor component type → (numpy dtype, byte width per scalar, struct.unpack flag).
_COMPONENT_INFO: dict[int, tuple[str, int, str]] = {
    5120: ("<i1", 1, "b"),  # BYTE
    5121: ("<u1", 1, "B"),  # UNSIGNED_BYTE
    5122: ("<i2", 2, "h"),  # SHORT
    5123: ("<u2", 2, "H"),  # UNSIGNED_SHORT
    5125: ("<u4", 4, "I"),  # UNSIGNED_INT
    5126: ("<f4", 4, "f"),  # FLOAT
}
_TYPE_COMPONENT_COUNT: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def load_glb(
    path: str | Path,
    *,
    target_fps: float = 60.0,
    target_up_axis: str = "Z",
    animation_index: int = 0,
    joint_names: list[str] | None = None,
    with_mesh: bool = False,
    simplify_rig: bool = True,
    progress_callback: Any = None,  # noqa: ARG001 — accepted for forward compatibility
) -> Motion:
    """Load a ``.glb`` / ``.gltf`` file into a :class:`Motion`.

    Args:
        path: Path to the file on disk.  Binary ``.glb`` and JSON ``.gltf`` (with sibling
            ``.bin`` / embedded data URIs) both work.
        target_fps: Uniform sample rate of the output motion.  Animation keyframes are
            resampled to this rate via slerp (rotation) or linear (translation / scale).
        target_up_axis: Axis that should be treated as "up" in the returned Motion.
            glTF natively uses ``"Y"``; set to ``"Z"`` (the default) to match the rest of
            the hhtools internal convention.
        animation_index: Which animation to import when the file contains more than one.
        joint_names: Optional whitelist of node names to keep in the output Motion.
            When provided, the hierarchy is re-rooted so that each listed name's parent
            is the nearest ancestor that is also on the whitelist.  When omitted, the
            first ``skins[*].joints`` list is used as the joint set — this matches what
            Mixamo, Maya, Blender, and Unity export by default.
        with_mesh: When True, also extract the first skin's skinned-mesh geometry into
            a :class:`~hhtools.core.skinning.SkinnedMesh` and attach it to
            ``Motion.meta["skinned_mesh"]``.  Vertex-buffer inverse-bind matrices are
            kept in the source (Y-up) frame; the Z-up rotation of joint transforms and
            the (identity) Y-up inverse-bind frame cancel at LBS time so that deformed
            vertices land in the Motion's current up-axis without extra conversion.
            Requires ``joint_names=None`` (i.e. the default skin-joints set) so that
            per-vertex ``JOINTS_0`` values remain valid Motion-bone indices.
        simplify_rig: When True (default), post-process the loaded Motion with
            :func:`hhtools.core.simplify.simplify_motion` to drop authoring-only
            helper bones (twist correctors, finger phalanges, face internals, end-tip
            ``_null`` placeholders, etc.).  This reduces the bone count from ~150 down
            to ~20–40 for typical DCC rigs (Cranberry 159 → 22) and keeps the viewer /
            retargeter focused on the anatomically meaningful subset.  Mesh weights for
            dropped bones are redistributed onto their nearest kept ancestor, so the
            LBS surface stays visually attached — fingers just lose their independent
            motion.  Set to False to keep the full rig (e.g. for debugging or when a
            downstream consumer needs every authoring bone).

    Raises:
        ValueError: if the file has no animations, the selected animation has no
            translation/rotation channels for any joint node, or the file has neither a
            skin nor an explicit ``joint_names`` whitelist.
        IndexError: if ``animation_index`` is out of range for the file.
        LookupError: if ``joint_names`` references a name not present in the glTF node
            list (clearer than the default KeyError we'd otherwise surface).

    Returns:
        A :class:`Motion` holding global joint positions + xyzw quaternions, plus metadata
        about the source animation, interpolation fallbacks, and any non-uniform scale
        approximations applied.
    """
    try:
        import pygltflib  # type: ignore[import-not-found]
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "GLB import requires the optional 'formats' extra: "
            "`pip install 'hhtools[formats]'` (installs pygltflib)."
        ) from err

    path = Path(path)
    gltf = pygltflib.GLTF2().load(str(path))
    # Load any embedded GLB binary chunk into the in-memory GLTF object so that
    # ``get_data_from_buffer_uri(None)`` works downstream.  This is a no-op when the file
    # is a .gltf with data URI / external bin.
    try:
        gltf.convert_images(pygltflib.ImageFormat.DATAURI, override=False)
    except Exception:  # noqa: BLE001  (pygltflib raises assorted errors on already-decoded files)
        pass

    if not gltf.animations:
        raise ValueError(f"GLB file {path} has no animations; cannot produce a Motion.")
    if animation_index < 0 or animation_index >= len(gltf.animations):
        raise IndexError(
            f"animation_index={animation_index} out of range for "
            f"{len(gltf.animations)} animations in {path}"
        )
    animation = gltf.animations[animation_index]

    node_table = _build_node_table(gltf)

    joint_node_indices, used_skin = _resolve_joint_selection(
        gltf, joint_node_indices_cache=None, joint_names=joint_names
    )
    if not joint_node_indices:
        raise ValueError(
            f"GLB file {path} has no skin and no joint_names whitelist provided."
        )

    # Sample every *full-tree* node's TRS at the uniform target_fps grid so that
    # non-joint ancestors with animated scale / translation still contribute to FK.
    sampled = _sample_all_nodes(gltf, animation, node_table, target_fps=target_fps)

    # Quaternion FK over full node tree, with parent-chain scale baked into child
    # translations.  Pure rotation quaternions are propagated by Q.multiply.
    global_pos_full, global_rot_full = _full_tree_fk(node_table, sampled)

    # Index out the joint subset and rebuild a hierarchy whose parent indices live
    # inside the subset (non-joint ancestors collapse to the nearest joint ancestor).
    positions, quaternions, hierarchy = _extract_joint_subset(
        node_table=node_table,
        joint_node_indices=joint_node_indices,
        global_pos=global_pos_full,
        global_rot=global_rot_full,
    )

    # glTF is Y-up; rotate into the requested internal up-axis (usually Z).
    if target_up_axis.upper() != "Y":
        rot_mat = R.up_axis_rotation("Y", target_up_axis.upper())
        positions = positions @ rot_mat.T
        rot_quat = Q.from_matrix(rot_mat)
        quaternions = Q.multiply(np.broadcast_to(rot_quat, quaternions.shape), quaternions)

    meta: dict[str, Any] = {
        "source_path": str(path),
        "source_up_axis": "Y",
        "animation_name": animation.name or f"animation_{animation_index}",
        "animation_index": animation_index,
        "num_animations": len(gltf.animations),
        "has_skin": bool(gltf.skins),
        "selected_via": "joint_names" if joint_names is not None else (
            "skins[0].joints" if used_skin else "none"
        ),
        "num_joints": len(joint_node_indices),
        "num_nodes_in_file": len(gltf.nodes),
    }
    if sampled.cubicspline_fallback:
        meta["cubicspline_fallback"] = sampled.cubicspline_fallback
    if sampled.raw_framerate is not None:
        meta["raw_framerate"] = sampled.raw_framerate
    if sampled.raw_duration_sec is not None:
        meta["raw_duration_sec"] = sampled.raw_duration_sec
    if sampled.scale_nonuniform:
        meta["scale_nonuniform"] = True
    if sampled.skeleton_warnings:
        meta["skeleton_warnings"] = sampled.skeleton_warnings

    if with_mesh:
        if joint_names is not None:
            raise ValueError(
                "load_glb(with_mesh=True) requires joint_names=None. "
                "Filtering joints to a whitelist invalidates the per-vertex JOINTS_0 "
                "indices baked into the skinned mesh (they reference the full skin), "
                "so the LBS blend would silently pull from the wrong bones."
            )
        # Mesh extraction is best-effort: the skeleton is always the primary
        # payload, and a mesh failure (missing skin / corrupt primitive /
        # JOINTS_0 indices out of range / …) must never prevent the skeleton
        # from loading.  We log a structured warning into ``meta`` and fall
        # back to capsule rendering on the viewer side, which already guards
        # on ``SkinnedMeshRenderer.has_mesh()``.
        try:
            skinned = _extract_skinned_mesh(
                gltf,
                joint_node_indices=joint_node_indices,
                num_motion_joints=len(joint_node_indices),
            )
        except Exception as mesh_err:  # noqa: BLE001 — intentional catch-all
            skinned = None
            meta["skinned_mesh_unavailable"] = True
            meta["skinned_mesh_error"] = f"{type(mesh_err).__name__}: {mesh_err}"
        if skinned is not None:
            meta["skinned_mesh"] = skinned
            meta["skinned_mesh_vertices"] = int(skinned.num_vertices)
            meta["skinned_mesh_triangles"] = int(skinned.num_triangles)
        else:
            # Skeleton-only asset or a soft-failure above — either way the
            # user asked for mesh but we can't honour it; flag the request so
            # the viewer can optionally toast "mesh unavailable, showing
            # capsules" instead of silently ignoring the toggle.
            meta.setdefault("skinned_mesh_unavailable", True)

    motion = Motion(
        name=path.stem,
        hierarchy=hierarchy,
        positions=positions.astype(np.float32),
        quaternions=quaternions.astype(np.float32),
        framerate=float(target_fps),
        up_axis=target_up_axis.upper(),  # type: ignore[arg-type]
        source_format="glb",
        meta=meta,
    )

    if simplify_rig:
        # Drop twist correctors, fingers, face internals, etc.  Runs after Motion
        # construction so the simplify pass reuses the same validated Hierarchy +
        # SkinnedMesh invariants every other caller sees.
        from hhtools.core.simplify import simplify_motion

        motion = simplify_motion(motion)
    return motion


# ---------------------------------------------------------------------- node table


class _NodeTable:
    """Flat view of every ``gltf.nodes[*]`` with parent pointers + rest TRS."""

    __slots__ = (
        "rest_rot",
        "rest_scale",
        "rest_trans",
        "name_to_index",
        "names",
        "num_nodes",
        "parent_of",
    )

    def __init__(
        self,
        names: list[str],
        parent_of: NDArray,
        rest_trans: NDArray,
        rest_rot: NDArray,
        rest_scale: NDArray,
    ) -> None:
        self.names = names
        self.parent_of = parent_of  # (N,) int, -1 for root
        self.rest_trans = rest_trans  # (N, 3)
        self.rest_rot = rest_rot  # (N, 4) xyzw
        self.rest_scale = rest_scale  # (N, 3)
        self.num_nodes = len(names)
        self.name_to_index = {n: i for i, n in enumerate(names)}


def _build_node_table(gltf) -> _NodeTable:  # type: ignore[no-untyped-def]
    """Collect every node's name, parent, and rest TRS into a flat table.

    We resolve the parent pointer by scanning each node's ``children``; glTF doesn't
    store it directly.  Nodes without a parent are roots (parent = -1).  A node's rest
    TRS is either the ``(translation, rotation, scale)`` triple or — when ``matrix`` is
    used instead — decomposed into TRS via column-norm scale extraction followed by
    ``Q.from_matrix`` on the orthonormalised rotation block.
    """
    num_nodes = len(gltf.nodes)
    names = [gltf.nodes[i].name or f"node_{i}" for i in range(num_nodes)]

    parent_of = np.full(num_nodes, -1, dtype=np.int32)
    for parent_idx, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parent_of[int(child)] = parent_idx

    rest_trans = np.zeros((num_nodes, 3), dtype=np.float32)
    rest_rot = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_nodes, 1)
    )  # xyzw identity
    rest_scale = np.ones((num_nodes, 3), dtype=np.float32)

    for i, node in enumerate(gltf.nodes):
        if node.matrix:
            mat = np.asarray(node.matrix, dtype=np.float32).reshape(4, 4).T  # col-major → row
            rest_trans[i] = mat[:3, 3]
            sx = float(np.linalg.norm(mat[:3, 0]))
            sy = float(np.linalg.norm(mat[:3, 1]))
            sz = float(np.linalg.norm(mat[:3, 2]))
            rest_scale[i] = [sx, sy, sz]
            norm_mat = mat[:3, :3].copy()
            if sx > 1e-8:
                norm_mat[:, 0] /= sx
            if sy > 1e-8:
                norm_mat[:, 1] /= sy
            if sz > 1e-8:
                norm_mat[:, 2] /= sz
            rest_rot[i] = Q.from_matrix(norm_mat)
        else:
            if node.translation:
                rest_trans[i] = node.translation
            if node.rotation:
                rest_rot[i] = node.rotation  # glTF quaternions are already xyzw
            if node.scale:
                rest_scale[i] = node.scale

    return _NodeTable(
        names=names,
        parent_of=parent_of,
        rest_trans=rest_trans,
        rest_rot=rest_rot,
        rest_scale=rest_scale,
    )


def _resolve_joint_selection(
    gltf,  # type: ignore[no-untyped-def]
    joint_node_indices_cache: list[int] | None,
    joint_names: list[str] | None,
) -> tuple[list[int], bool]:
    """Pick which glTF node indices become the output hierarchy's joints.

    Returns ``(indices, used_skin)``.  ``used_skin`` is True only when the indices came
    from ``skins[0].joints`` — that way the caller can distinguish an explicit
    ``joint_names`` query from the default skin-driven path in the returned meta dict.
    """
    if joint_names is not None:
        node_name_to_idx: dict[str, int] = {
            (n.name or f"node_{i}"): i for i, n in enumerate(gltf.nodes)
        }
        missing = [name for name in joint_names if name not in node_name_to_idx]
        if missing:
            raise LookupError(
                f"joint_names references {len(missing)} name(s) not found in glTF: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return [node_name_to_idx[name] for name in joint_names], False

    if joint_node_indices_cache is not None:
        return joint_node_indices_cache, True

    if gltf.skins:
        joints = list(gltf.skins[0].joints or [])
        if joints:
            return [int(j) for j in joints], True

    return [], False


# ---------------------------------------------------------------------- sampling


class _SampledAnimation:
    """Per-frame TRS for every glTF node after uniform-grid resampling."""

    __slots__ = (
        "local_rot",
        "local_scale",
        "local_trans",
        "cubicspline_fallback",
        "num_frames",
        "raw_duration_sec",
        "raw_framerate",
        "scale_nonuniform",
        "skeleton_warnings",
    )

    def __init__(
        self,
        local_trans: NDArray,
        local_rot: NDArray,
        local_scale: NDArray,
        num_frames: int,
        cubicspline_fallback: list[str],
        scale_nonuniform: bool,
        raw_framerate: float | None,
        raw_duration_sec: float | None,
        skeleton_warnings: list[str],
    ) -> None:
        self.local_trans = local_trans  # (F, N, 3)
        self.local_rot = local_rot  # (F, N, 4)
        self.local_scale = local_scale  # (F, N, 3)
        self.num_frames = num_frames
        self.cubicspline_fallback = cubicspline_fallback
        self.scale_nonuniform = scale_nonuniform
        self.raw_framerate = raw_framerate
        self.raw_duration_sec = raw_duration_sec
        self.skeleton_warnings = skeleton_warnings


def _sample_all_nodes(
    gltf,  # type: ignore[no-untyped-def]
    animation,  # type: ignore[no-untyped-def]
    node_table: _NodeTable,
    *,
    target_fps: float,
) -> _SampledAnimation:
    """Resample every animation channel onto a uniform ``target_fps`` time grid.

    Nodes without an animation channel for a given TRS component hold at their rest
    value across all frames.  ``CUBICSPLINE`` tracks are reduced to their middle
    "value" row per keyframe (tangents discarded) — each fallback is recorded as a
    ``"nodeN.TYPE"`` string so callers can surface the list to the user.

    ``raw_framerate`` / ``raw_duration_sec`` are derived from the animation's own input
    timestamps (assumed equispaced, which is the case for every glTF file we've seen in
    the wild); they're stored as provenance in Motion.meta.
    """
    path_to_key = {"TRANSLATION": "T", "ROTATION": "R", "SCALE": "S"}
    all_times: list[float] = []
    # channel_store[node_idx][key] = (times, values, interp)
    channel_store: list[dict[str, tuple[NDArray, NDArray, str]]] = [
        {} for _ in range(node_table.num_nodes)
    ]
    first_input_times: NDArray | None = None

    for ch in animation.channels:
        if ch.target is None:
            continue
        node_idx = ch.target.node
        if node_idx is None or node_idx < 0 or node_idx >= node_table.num_nodes:
            continue
        sampler = animation.samplers[ch.sampler]
        times = _read_accessor(gltf, sampler.input).astype(np.float32).reshape(-1)
        if first_input_times is None:
            first_input_times = times
        values = _read_accessor(gltf, sampler.output).astype(np.float32)
        interp = (sampler.interpolation or "LINEAR").upper()
        key = path_to_key.get((ch.target.path or "").upper())
        if key is None:
            continue  # Ignore morph-target WEIGHTS.
        channel_store[int(node_idx)][key] = (times, values, interp)
        all_times.extend(times.tolist())

    if not all_times:
        raise ValueError("Animation has no translation / rotation / scale channels.")

    t_min = float(min(all_times))
    t_max = float(max(all_times))
    duration = max(t_max - t_min, 1.0 / float(target_fps))
    num_frames = max(int(round(duration * float(target_fps))) + 1, 1)
    times_out = t_min + np.arange(num_frames, dtype=np.float32) / float(target_fps)
    times_out = np.clip(times_out, t_min, t_max)

    # Preallocate: rest values broadcast across all frames; channels overwrite where present.
    local_trans = np.broadcast_to(
        node_table.rest_trans[None, :, :], (num_frames, node_table.num_nodes, 3)
    ).copy()
    local_rot = np.broadcast_to(
        node_table.rest_rot[None, :, :], (num_frames, node_table.num_nodes, 4)
    ).copy()
    local_scale = np.broadcast_to(
        node_table.rest_scale[None, :, :], (num_frames, node_table.num_nodes, 3)
    ).copy()

    cubic_fallback: list[str] = []
    for node_idx in range(node_table.num_nodes):
        for key, (t_key, v_key, interp) in channel_store[node_idx].items():
            used_interp = interp
            if interp == "CUBICSPLINE":
                per_key = v_key.reshape(-1, 3, v_key.size // (len(t_key) * 3))
                v_key = per_key[:, 1, :]  # middle (value) row
                used_interp = "LINEAR"
                cubic_fallback.append(f"{node_table.names[node_idx]}.{key}")
            if key == "R":
                local_rot[:, node_idx] = _sample_rotation(times_out, t_key, v_key, used_interp)
            elif key == "T":
                local_trans[:, node_idx] = _sample_vector(times_out, t_key, v_key, used_interp)
            else:  # "S"
                local_scale[:, node_idx] = _sample_vector(times_out, t_key, v_key, used_interp)

    scale_nonuniform = bool(np.any(np.abs(local_scale - 1.0) > 1e-4))

    raw_framerate = None
    raw_duration = None
    if first_input_times is not None and first_input_times.size >= 2:
        raw_duration = float(first_input_times[-1] - first_input_times[0])
        if raw_duration > 0:
            raw_framerate = float((first_input_times.size - 1) / raw_duration)

    return _SampledAnimation(
        local_trans=local_trans,
        local_rot=local_rot,
        local_scale=local_scale,
        num_frames=num_frames,
        cubicspline_fallback=cubic_fallback,
        scale_nonuniform=scale_nonuniform,
        raw_framerate=raw_framerate,
        raw_duration_sec=raw_duration,
        skeleton_warnings=[],
    )


def _sample_vector(t_out: NDArray, t_key: NDArray, v_key: NDArray, interp: str) -> NDArray:
    """Sample a vec3 track (translation or scale).

    LINEAR uses per-component ``np.interp``; STEP uses left-constant nearest-keyframe.
    """
    t_out = np.asarray(t_out, dtype=np.float32).reshape(-1)
    t_key = np.asarray(t_key, dtype=np.float32).reshape(-1)
    v_key = np.asarray(v_key, dtype=np.float32).reshape(len(t_key), -1)
    if interp == "STEP":
        idx = np.searchsorted(t_key, t_out, side="right") - 1
        idx = np.clip(idx, 0, len(t_key) - 1)
        return v_key[idx]
    out = np.empty((t_out.size, v_key.shape[1]), dtype=np.float32)
    for k in range(v_key.shape[1]):
        out[:, k] = np.interp(t_out, t_key, v_key[:, k])
    return out


def _sample_rotation(t_out: NDArray, t_key: NDArray, v_key: NDArray, interp: str) -> NDArray:
    """Sample a quaternion track using slerp (LINEAR) or step (STEP)."""
    t_out = np.asarray(t_out, dtype=np.float32).reshape(-1)
    t_key = np.asarray(t_key, dtype=np.float32).reshape(-1)
    v_key = np.asarray(v_key, dtype=np.float32).reshape(len(t_key), 4)
    if interp == "STEP":
        idx = np.searchsorted(t_key, t_out, side="right") - 1
        idx = np.clip(idx, 0, len(t_key) - 1)
        return Q.normalize(v_key[idx])
    if len(t_key) == 1:
        return Q.normalize(np.broadcast_to(v_key[0], (t_out.size, 4)).copy())
    idx1 = np.searchsorted(t_key, t_out, side="left")
    idx1 = np.clip(idx1, 1, len(t_key) - 1)
    idx0 = idx1 - 1
    t0 = t_key[idx0]
    t1 = t_key[idx1]
    dt = np.maximum(t1 - t0, 1e-8)
    u = np.clip((t_out - t0) / dt, 0.0, 1.0).astype(np.float32)
    return Q.slerp(v_key[idx0], v_key[idx1], u)


# ---------------------------------------------------------------------- FK


def _full_tree_fk(
    node_table: _NodeTable, sampled: _SampledAnimation
) -> tuple[NDArray, NDArray]:
    """Forward kinematics over every glTF node.

    Scale propagation strategy (same shape as ai4animationpy.GLBImporter but ported to
    hhtools' quaternion-centric math):

    1. Compute each node's *parent-chain-accumulated* scale (element-wise product).
       The root's effective parent scale is 1; child's is ``parent_scale * parent_own_scale``.
    2. Bake that parent scale into the child's local translation (``t *= parent_scale``).
       Root translations are left untouched — they live in the world frame already.
    3. Run standard quaternion FK on ``(scaled_local_trans, local_rot)``.

    Returns ``(global_pos (F, N, 3), global_rot (F, N, 4))``.  The rotation channel is
    *pure* (scale is not embedded) because we baked scale into translation and never into
    the quaternion.  This matches the semantics of ``Motion.quaternions``.
    """
    F = sampled.num_frames
    N = node_table.num_nodes

    parent_scale = np.ones((F, N, 3), dtype=np.float32)
    scaled_trans = np.empty((F, N, 3), dtype=np.float32)
    global_pos = np.empty((F, N, 3), dtype=np.float32)
    global_rot = np.empty((F, N, 4), dtype=np.float32)

    # Iterate in node order — glTF doesn't guarantee topological ordering, but nodes are
    # conventionally emitted in top-down DFS (parent before children) by all mainstream
    # exporters.  We don't rely on it: if a parent hasn't been processed yet we fall back
    # to raw parent-chain walking.  To make this robust, we just sort by depth first.
    order = _topological_order(node_table.parent_of)
    for i in order:
        p = int(node_table.parent_of[i])
        if p == -1:
            parent_scale[:, i] = 1.0  # root: no parent scale
            scaled_trans[:, i] = sampled.local_trans[:, i]
            global_pos[:, i] = scaled_trans[:, i]
            global_rot[:, i] = sampled.local_rot[:, i]
        else:
            parent_scale[:, i] = parent_scale[:, p] * sampled.local_scale[:, p]
            scaled_trans[:, i] = sampled.local_trans[:, i] * parent_scale[:, i]
            global_rot[:, i] = Q.multiply(global_rot[:, p], sampled.local_rot[:, i])
            global_pos[:, i] = global_pos[:, p] + Q.rotate(global_rot[:, p], scaled_trans[:, i])

    return global_pos, global_rot


def _topological_order(parent_of: NDArray) -> list[int]:
    """Return node indices sorted parent-before-child using a depth count.

    A node's depth is ``depth[parent] + 1`` (0 for roots); sorting by depth guarantees
    the FK loop always sees a parent before its descendants without an explicit DFS.
    """
    N = parent_of.shape[0]
    depth = np.zeros(N, dtype=np.int32)
    for i in range(N):
        p = int(parent_of[i])
        while p != -1:
            depth[i] += 1
            p = int(parent_of[p])
    return np.argsort(depth, kind="stable").tolist()


# ---------------------------------------------------------------------- subset + hierarchy


def _extract_joint_subset(
    node_table: _NodeTable,
    joint_node_indices: list[int],
    global_pos: NDArray,
    global_rot: NDArray,
) -> tuple[NDArray, NDArray, Hierarchy]:
    """Index out the joints we actually want to expose, and rebuild their hierarchy.

    The parent of joint ``j`` in the output hierarchy is the nearest glTF-node ancestor
    of ``j`` that is *also in the joint list* — or -1 if no such ancestor exists.  This
    matches how the skin is drawn in every authoring tool: non-joint helper nodes between
    two joints are skipped, but their effect on joint positions is already captured by
    the full-tree FK above.
    """
    node_is_joint = {n: i for i, n in enumerate(joint_node_indices)}
    num_joints = len(joint_node_indices)

    bone_names: list[str] = []
    parent_indices = np.full(num_joints, -1, dtype=np.int32)
    seen_names: set[str] = set()

    for j, node_idx in enumerate(joint_node_indices):
        base_name = node_table.names[node_idx]
        name = base_name
        # Dedup — glTF allows duplicate node names but Hierarchy requires uniqueness.
        counter = 0
        while name in seen_names:
            counter += 1
            name = f"{base_name}__{counter}"
        seen_names.add(name)
        bone_names.append(name)

        p = int(node_table.parent_of[node_idx])
        while p != -1 and p not in node_is_joint:
            p = int(node_table.parent_of[p])
        parent_indices[j] = node_is_joint.get(p, -1)

    hierarchy = Hierarchy.from_parent_indices(bone_names, parent_indices)
    positions = global_pos[:, joint_node_indices, :]
    quaternions = global_rot[:, joint_node_indices, :]
    return positions, quaternions, hierarchy


# ---------------------------------------------------------------------- skinned mesh


def _extract_skinned_mesh(
    gltf,  # type: ignore[no-untyped-def]
    *,
    joint_node_indices: list[int],
    num_motion_joints: int,
) -> SkinnedMesh | None:
    """Pull ``skins[0]`` inverseBindMatrices + every node referencing that skin's mesh
    primitives into a single :class:`SkinnedMesh`.

    What gets concatenated and why:

    * We walk every ``gltf.nodes[*]`` whose ``skin`` matches the skin we picked for the
      hierarchy.  A character file may split its mesh across primitives (for different
      materials — hair, body, eyes) or across separate mesh nodes that share the same
      skin (rare but valid).  We merge them because the renderer currently treats the
      character as one colour / one material anyway, and an inconsistent per-primitive
      split would mean the renderer has to juggle multiple handles per frame.

    * Per-vertex ``JOINTS_0`` are indices into ``skin.joints`` which, by our
      :func:`_resolve_joint_selection` default, are also the Motion bone indices (we
      pass the joint list through in its original order).  That's why
      :attr:`SkinnedMesh.skin_indices` can be stored as-is without a re-mapping table.

    * If a node's ``gltf.nodes[*].skin`` points at a different skin than the one we used
      for the hierarchy we skip it — otherwise we'd end up with two vertex groups whose
      bone indices reference different joint sets.  The number of skipped meshes is
      surfaced in ``Motion.meta["skinned_mesh_skipped_nodes"]`` by the caller when
      non-zero so users can see the truncation.

    Returns None when the file has no skin at all (e.g. legacy rigs imported as raw
    joint hierarchies).  A ValueError is raised when a skin exists but every mesh
    primitive turned out to be incompatible (wrong skin / missing POSITION / …) — silent
    empty meshes would be harder to debug downstream.
    """
    if not gltf.skins:
        return None

    skin_idx_used = _find_skin_matching(gltf, joint_node_indices)
    if skin_idx_used is None:
        # A skin exists but our hierarchy came from an explicit whitelist that doesn't
        # line up with any skin.joints list — we'd need a re-mapping table to salvage
        # the mesh, which isn't worth the complexity for v1.
        return None
    skin = gltf.skins[skin_idx_used]

    # Inverse-bind matrices are glTF column-major; reshape then transpose per-joint.
    if skin.inverseBindMatrices is None:
        raise ValueError(
            f"Skin {skin_idx_used} has no inverseBindMatrices accessor — cannot "
            "perform LBS without per-joint bind-pose data."
        )
    ibm_raw = _read_accessor(gltf, skin.inverseBindMatrices).astype(np.float32)
    expected_joints = len(skin.joints or [])
    ibm = ibm_raw.reshape(expected_joints, 4, 4).transpose(0, 2, 1)

    # If the Motion carries fewer joints than the skin (shouldn't happen in the default
    # path, but a defensive check keeps the viewer from indexing out of bounds later),
    # we pad with identity and note the mismatch.
    if num_motion_joints < expected_joints:
        pad = np.broadcast_to(
            np.eye(4, dtype=np.float32),
            (expected_joints - num_motion_joints, 4, 4),
        )
        ibm = np.concatenate([ibm[:num_motion_joints], pad], axis=0)
    elif num_motion_joints > expected_joints:
        pad = np.broadcast_to(
            np.eye(4, dtype=np.float32),
            (num_motion_joints - expected_joints, 4, 4),
        )
        ibm = np.concatenate([ibm, pad], axis=0)

    all_verts: list[NDArray] = []
    all_tris: list[NDArray] = []
    all_joints: list[NDArray] = []
    all_weights: list[NDArray] = []
    vert_offset = 0
    # Track whether the skin is even *referenced* by any mesh node.  Pure-skeleton
    # assets (e.g. a mocap-only FBX exported with the armature but no body geometry)
    # will have ``gltf.skins`` populated but zero mesh nodes pointing at them; we
    # handle this as a graceful "no mesh available" (return None) rather than an
    # error, so callers that set ``with_mesh=True`` as a UI preference still get a
    # working skeleton when the asset happens to lack a body.
    skin_referenced_by_mesh_node = False

    for node in gltf.nodes:
        if node.skin is None or node.skin != skin_idx_used:
            continue
        if node.mesh is None:
            continue
        skin_referenced_by_mesh_node = True
        mesh = gltf.meshes[node.mesh]
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue  # Pure-morph mesh without rest pose; skip.
            if prim.attributes.JOINTS_0 is None or prim.attributes.WEIGHTS_0 is None:
                # A non-skinned primitive baked into a skinned mesh node (rare; some
                # Mixamo eye rigs do this).  Without joint/weight data we can't LBS it,
                # so we skip it rather than silently staple it to joint 0.
                continue

            pos = _read_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
            joints = _read_accessor(gltf, prim.attributes.JOINTS_0).astype(np.int32)
            weights = _read_accessor(gltf, prim.attributes.WEIGHTS_0).astype(np.float32)

            # Triangle indices: ``prim.indices`` may be None (draw-array style); in that
            # case vertex i triple (3i, 3i+1, 3i+2) forms a triangle.
            if prim.indices is not None:
                idx = _read_accessor(gltf, prim.indices).astype(np.int32).reshape(-1)
                tris = idx.reshape(-1, 3)
            else:
                v_count = pos.shape[0]
                tris = np.arange(v_count, dtype=np.int32).reshape(-1, 3)

            all_verts.append(pos)
            all_tris.append(tris + vert_offset)
            all_joints.append(joints)
            all_weights.append(weights)
            vert_offset += pos.shape[0]

    if not all_verts:
        if not skin_referenced_by_mesh_node:
            # Asset is skeleton-only (common for mocap FBX exports, Cranberry rig,
            # etc.).  Return None so the caller falls back to capsule rendering
            # without raising — the skeleton still loads fine.
            return None
        raise ValueError(
            f"Skin {skin_idx_used} has mesh nodes attached but none of their "
            "primitives carry POSITION + JOINTS_0 + WEIGHTS_0 — the file is "
            "likely corrupt or was exported with skin data stripped."
        )

    verts_cat = np.concatenate(all_verts, axis=0)
    tris_cat = np.concatenate(all_tris, axis=0).astype(np.int32)
    joints_cat = np.concatenate(all_joints, axis=0)
    weights_cat = np.concatenate(all_weights, axis=0)

    return SkinnedMesh(
        vertices_rest=verts_cat,
        triangles=tris_cat,
        skin_indices=joints_cat,
        skin_weights=weights_cat,
        inverse_bind=ibm,
    )


def _find_skin_matching(gltf, joint_node_indices: list[int]) -> int | None:  # type: ignore[no-untyped-def]
    """Return the skin whose ``joints`` list equals ``joint_node_indices`` (as a set).

    We compare as a set (not a sequence) because the Motion's hierarchy preserves
    order but the mesh primitive's ``JOINTS_0`` remaps through ``skin.joints`` — what
    matters is that every index emitted from the mesh can be looked up in the joint
    list, not that the two lists are in identical order.  When multiple skins share the
    same joint set we just pick the first; real-world files almost never have more than
    one skin anyway.
    """
    want = set(int(i) for i in joint_node_indices)
    for k, skin in enumerate(gltf.skins or []):
        if skin.joints is None:
            continue
        if set(int(j) for j in skin.joints) == want:
            return k
    return None


# ---------------------------------------------------------------------- buffer IO


def _read_accessor(gltf, accessor_index: int) -> NDArray:  # type: ignore[no-untyped-def]
    """Read an accessor into a dense numpy array of shape ``(count, comp_count)``.

    We delegate buffer resolution to ``pygltflib.GLTF2.get_data_from_buffer_uri`` so that
    embedded GLB chunks, ``data:`` URIs, and external ``.bin`` siblings all work through
    one code path.  Interleaved buffer views (``byteStride > component_count * byte_width``)
    are unpacked explicitly into a contiguous row-major array.  SCALAR accessors collapse
    to shape ``(count,)`` — this saves a squeeze at every call site.
    """
    acc = gltf.accessors[accessor_index]
    view = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[view.buffer]

    data = gltf.get_data_from_buffer_uri(buf.uri)
    dtype_str, elem_bytes, struct_flag = _COMPONENT_INFO[acc.componentType]
    comp_count = _TYPE_COMPONENT_COUNT[acc.type]
    count = acc.count
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    stride = view.byteStride or (elem_bytes * comp_count)
    natural_stride = elem_bytes * comp_count

    if stride == natural_stride:
        byte_slice = data[offset : offset + count * natural_stride]
        arr = np.frombuffer(byte_slice, dtype=np.dtype(dtype_str))
        arr = arr.reshape(count, comp_count)
    else:
        # Interleaved: decode row-by-row using struct.unpack to honour arbitrary stride.
        arr = np.empty((count, comp_count), dtype=dtype_str)
        row_flag = "<" + struct_flag * comp_count
        for k in range(count):
            row_off = offset + k * stride
            arr[k] = struct.unpack(row_flag, data[row_off : row_off + natural_stride])

    if comp_count == 1:
        arr = arr.reshape(count)
    return np.asarray(arr)


__all__ = ["load_glb"]
