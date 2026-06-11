"""Skeleton simplification: prune non-anatomical and auxiliary bones from a Motion.

Authoring rigs (Maya / Blender / character artist pipelines) ship with dozens of helper
joints that have no direct anatomical counterpart — twist correction bones, muscle
drivers, end-tip "null" nodes, individual finger phalanges, eye / jaw / tongue bones.
Cranberry (GLB) has 159 joints for a character whose SMPL equivalent is 23.  These
extras *are* necessary for high-fidelity skin deformation authored in the DCC, but for
retargeting, visualisation, and math work we almost never want to reason about them:

* retargeters pair up human and robot joints at the "body / limb / palm / head"
  granularity; fingers and face get handled by separate pipelines if at all,
* capsule / skeleton viewers look cluttered and deceiving at 150+ bones,
* CSV / trajectory exports balloon in size with mostly-constant twist values.

:func:`simplify_motion` keeps only the bones that survive a caller-supplied predicate,
remaps the parent chain (so dropped bones' children reparent to the nearest kept
ancestor), and — importantly — rebuilds any attached :class:`SkinnedMesh` so skin
weights for dropped bones flow to the nearest kept ancestor.  The mesh stays fully
attached (no floating verts), it just loses the fine detail those extra bones provided.

The default predicate :func:`default_keep_predicate` targets the "body + limbs + palms
+ head" subset the viewer wants: it drops fingers, face internals (eyes, jaw, tongue,
teeth, brows, lips), rig-helper prefixes (``p_``, ``helper_``, ``DEF_``) and suffixes
(``_null``, ``_end``, ``_nub``, ``_twist*``), foot tarsal subdivisions, and a
handful of common authoring wrappers already flagged as virtual roots.

All shape invariants (Motion.positions / quaternions framerate / up-axis /
source_format / meta) are preserved; only ``hierarchy`` changes size and any attached
``skinned_mesh`` is rebuilt.  BakedMesh caches are left untouched — they are vertex
clouds in world space with no per-bone attachment.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import numpy as np

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion

# Patterns matched against lowercase bone names (after stripping common namespace
# prefixes like ``Armature:``).  A bone is *dropped* if ANY pattern matches.
_DROP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        # Rig helper prefixes.
        r"^p_",         # Cranberry / Paragon twist + muscle helpers (p_r_arm_twist0, ...)
        r"^def_",       # Rigify deform helpers
        r"^mch_",       # Rigify mechanism bones
        r"^helper_",
        r"^ctrl_",      # rig controllers (not real bones)
        r"^ik_",
        # End-effector tips and padding.
        r"_null$",
        r"_end$",
        r"_nub$",
        r"_tip$",
        r"_target$",
        r"_tgt$",
        r"_pole$",
        # Twist distribution bones.
        r"_twist\d*$",
        r"_twist_",
        # Fingers.  We keep the wrist (hand/palm root) but drop per-finger phalanges.
        # We allow either a camel-case-friendly bare name (``LeftHandIndex1`` → ``...index1``)
        # or the more common snake-case (``b_l_index1``, ``l_thumb0``); ``(\d*)?`` lets the
        # trailing digit be optional (e.g. ``r_thumb_null``) and the right anchor
        # guarantees we don't mis-match ``indexing`` or similar non-finger tokens.
        r"(thumb|index|middle|ring|pinky)(\d+)?(_|$)",
        r"(^|_)finger\d*",
        # Face & head internals.
        r"(^|_)eye(\d*|ball|lid|brow)?(_|$)",
        r"(^|_)jaw(\d*)?(_|$)",
        r"(^|_)teeth(\d*)?(_|$)",
        r"(^|_)tongue\d*",
        r"(^|_)lip\d*",
        r"(^|_)cheek\d*",
        r"(^|_)nose\d*",
        r"(^|_)ear(\d*)?(_|$)",
        r"(^|_)brow\d*",
        r"(^|_)lid\d*",
        # Foot tarsal subdivisions and toe-ball end joints.  In authoring rigs like
        # Cranberry the ball lives on a *parallel* chain to the ankle (both are
        # children of ``foot_twist`` via ``talocrural → subtalar → transversetarsal``);
        # keeping the ball without those intermediates produces a "second lower leg"
        # drawn from the knee straight to the toe.  SMPL has no separate toe bone
        # either, so dropping ball keeps the foot as a clean leaf and matches the
        # retargeting target we care about.
        r"(^|_)talocrural(\d*)?(_|$)",
        r"(^|_)subtalar(\d*)?(_|$)",
        r"(^|_)transversetarsal(\d*)?(_|$)",
        r"(^|_)ball(\d*)?(_|$)",
        # Scapula / delt / glute muscle helpers (Cranberry-style p_* already caught;
        # this catches non-prefixed variants).
        r"^scap_",
        r"^delt_",
    )
)


def default_keep_predicate(name: str) -> bool:
    """Return ``True`` when *name* should be kept in the simplified skeleton.

    The check is case-insensitive and strips common authoring namespace prefixes
    (``Armature:``, ``mixamorig:``, ``metarig:``) before matching.  Returning ``False``
    means the bone is dropped; any vertices weighted to it will be redistributed onto
    its nearest kept ancestor by :func:`simplify_motion`.
    """
    stripped = name.split(":")[-1].lower()
    for pattern in _DROP_PATTERNS:
        if pattern.search(stripped):
            return False
    return True


def _nearest_kept_ancestor(
    old_idx: int, parents: list[int], kept: set[int]
) -> int:
    """Walk up ``parents`` from ``old_idx`` until we hit a kept bone (or root)."""
    cur = parents[old_idx]
    while cur >= 0 and cur not in kept:
        cur = parents[cur]
    return cur


def simplify_motion(
    motion: Motion,
    *,
    keep: Callable[[str], bool] | Iterable[str] | None = None,
) -> Motion:
    """Return a copy of *motion* restricted to bones matching *keep*.

    Args:
        motion: Source motion, optionally with ``meta["skinned_mesh"]``.
        keep: Either a predicate ``name -> bool`` or an explicit iterable of bone names
            to keep.  Defaults to :func:`default_keep_predicate` when ``None``.

    Returns:
        A new :class:`Motion` with:
            - ``hierarchy`` containing only the kept bones, parents remapped to new
              indices (dropped bones collapse into their nearest kept ancestor).
            - ``positions`` / ``quaternions`` trimmed on the bone axis.
            - ``meta["skinned_mesh"]`` rebuilt with dropped-bone weights reassigned
              to the nearest kept ancestor (mesh stays visually attached).
            - ``meta["simplify_dropped_bones"]`` recording the names of bones removed
              so downstream tooling (or "Show full rig" toggles) can reverse the
              operation by reloading from source.

    Raises:
        ValueError: if the predicate would drop all bones, or if the motion has no
            root (i.e. no bone with parent -1) after simplification.
    """
    bone_names = list(motion.hierarchy.bone_names)
    parents = [int(p) for p in motion.hierarchy.parent_indices]

    if keep is None:
        predicate: Callable[[str], bool] = default_keep_predicate
    elif callable(keep):
        predicate = keep  # type: ignore[assignment]
    else:
        whitelist = {str(n) for n in keep}
        predicate = lambda n: n in whitelist  # noqa: E731 — concise and local

    keep_idx: list[int] = [i for i, n in enumerate(bone_names) if predicate(n)]
    if not keep_idx:
        raise ValueError("simplify_motion would drop every bone; check the predicate")
    kept_set: set[int] = set(keep_idx)
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(keep_idx)}

    # Rebuild parents: parent(new_i) = new_index_of(nearest_kept_ancestor(old_i))
    new_parents: list[int] = []
    for old_i in keep_idx:
        anc = _nearest_kept_ancestor(old_i, parents, kept_set)
        new_parents.append(-1 if anc < 0 else old_to_new[anc])

    new_bone_names = [bone_names[i] for i in keep_idx]
    new_hierarchy = Hierarchy.from_parent_indices(new_bone_names, new_parents)

    new_positions = motion.positions[:, keep_idx, :].astype(np.float32, copy=True)
    new_quaternions = motion.quaternions[:, keep_idx, :].astype(np.float32, copy=True)

    dropped_bones = [bone_names[i] for i in range(len(bone_names)) if i not in kept_set]
    new_meta = dict(motion.meta)
    new_meta["simplify_dropped_bones"] = tuple(dropped_bones)

    # Re-attach the mesh, rewriting any skin_indices that pointed into dropped bones.
    skinned = new_meta.get("skinned_mesh")
    if skinned is not None:
        new_meta["skinned_mesh"] = _remap_skinned_mesh(
            skinned, keep_idx, parents, kept_set, old_to_new
        )

    return Motion(
        name=motion.name,
        hierarchy=new_hierarchy,
        positions=new_positions,
        quaternions=new_quaternions,
        framerate=motion.framerate,
        up_axis=motion.up_axis,
        source_format=motion.source_format,
        meta=new_meta,
        objects=motion.objects,
        terrain=motion.terrain,
    )


def _remap_skinned_mesh(
    mesh,  # SkinnedMesh — typed lazily to avoid import cycle
    keep_idx: list[int],
    parents: list[int],
    kept_set: set[int],
    old_to_new: dict[int, int],
):
    """Redistribute ``mesh.skin_indices`` onto kept bones (nearest ancestor)."""
    from hhtools.core.skinning import SkinnedMesh

    if not isinstance(mesh, SkinnedMesh):
        return mesh

    # Build old->new bone-index LUT.  For a dropped bone we fall back to its nearest
    # kept ancestor; for the pathological case of no kept ancestor (e.g. the predicate
    # dropped everything above the current joint and there's no root left) we point to
    # the new root (new index 0), keeping the mesh visually attached instead of
    # crashing on an out-of-range index.
    lut = np.zeros(len(parents), dtype=np.int32)
    for old_i in range(len(parents)):
        if old_i in kept_set:
            lut[old_i] = old_to_new[old_i]
        else:
            anc = _nearest_kept_ancestor(old_i, parents, kept_set)
            if anc >= 0:
                lut[old_i] = old_to_new[anc]
            else:
                lut[old_i] = 0  # root fallback; guarantees a valid index

    # Apply LUT.  Multiple influence slots of the same vertex may now alias to the
    # same new bone — e.g. all four finger phalanges collapsing onto the wrist — so we
    # merge their weights row-wise.  This keeps the total skin weight per vertex at 1
    # and cleans out dead (weight=0) slots.
    old_indices = mesh.skin_indices  # (V, K)
    old_weights = mesh.skin_weights
    v, k = old_indices.shape
    new_indices_raw = lut[old_indices]  # (V, K) but may contain duplicates

    # Row-merge duplicates.  We pick a fixed width K (same as source) which is always
    # enough since merging can only reduce the unique count per row.
    new_idx = np.zeros((v, k), dtype=np.int32)
    new_w = np.zeros((v, k), dtype=np.float32)
    for i in range(v):
        per_bone: dict[int, float] = {}
        for j in range(k):
            if old_weights[i, j] <= 0.0:
                continue
            b = int(new_indices_raw[i, j])
            per_bone[b] = per_bone.get(b, 0.0) + float(old_weights[i, j])
        # Sort by descending weight, keep up to K entries.
        top = sorted(per_bone.items(), key=lambda kv: -kv[1])[:k]
        for slot, (b, w) in enumerate(top):
            new_idx[i, slot] = b
            new_w[i, slot] = w
        # Renormalise (merging may leave the row fractionally under 1 only via
        # floating-point drift; the SkinnedMesh ctor also renormalises but we do it
        # here for clarity).
        s = new_w[i].sum()
        if s > 1e-8:
            new_w[i] /= s

    new_inverse_bind = mesh.inverse_bind[keep_idx].astype(np.float32, copy=True)
    return SkinnedMesh(
        vertices_rest=mesh.vertices_rest,
        triangles=mesh.triangles,
        skin_indices=new_idx,
        skin_weights=new_w,
        inverse_bind=new_inverse_bind,
        normals_rest=mesh.normals_rest,
    )


__all__ = ["default_keep_predicate", "simplify_motion"]
