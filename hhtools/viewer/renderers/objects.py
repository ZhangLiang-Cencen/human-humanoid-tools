"""Render :class:`SceneObject` trajectories into a Viser scene.

Two render paths live side by side:

* **Placeholder cuboid** — when ``SceneObject.mesh_path`` is empty, we create a single
  :meth:`viser.SceneApi.add_box` handle per object and rewrite its ``position`` / ``wxyz``
  every frame.
* **Real triangle mesh** — when ``mesh_path`` points to an existing file (``.obj`` / ``.ply`` /
  ``.stl`` — anything ``trimesh`` can read), we load the mesh once via :func:`trimesh.load`,
  scale the vertices by :attr:`SceneObject.scale`, register them with
  :meth:`viser.SceneApi.add_mesh_simple`, then update the handle's pose every frame. OMOMO's
  ``captured_objects/*.obj`` are the headline use case: drop them next to the clip's ``.pkl``
  and the adapter + this renderer pick them up automatically.

Both paths version-suffix handle names (``/objects/<name>_v{N}``) so an async remove-then-add
during motion switching can never leak the previous clip's geometry on top of the fresh one.

Mesh loading is guarded by a tiny in-process cache keyed on the absolute mesh path + scale, so
repeatedly selecting the same clip doesn't re-parse the OBJ from disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hhtools.core.scene import SceneObject

# Cache maps ``(resolved_path_str, scale_rounded)`` → (vertices, faces). We round the scale
# to 6 decimals so floating-point jitter between clip loads can't fragment the cache.
_MESH_CACHE: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

# Name-based visual policy. Rendering hints are *not* persisted into the NPZ (that file
# carries only trajectory + topology); the viewer applies the look here by matching
# the SceneObject's semantic name. Explicit per-object overrides via
# ``SceneObject.opacity`` / ``color`` still win — this dict is the fallback when those
# fields are ``None``.
#
# Terrain / ground / floor are intentionally **not** listed here: terrain now travels
# through the pipeline as a :class:`hhtools.core.scene.TerrainHeightfield` on
# :attr:`Motion.terrain`, rendered by :class:`TerrainHeightfieldRenderer` rather than
# this renderer.  Add per-prop entries (e.g. ``("box", ...)``) here if a future dataset
# wants a name-based default look.
_NAME_STYLE_RULES: tuple[tuple[str, tuple[int, int, int], float], ...] = ()


def _style_from_name(name: str) -> tuple[tuple[int, int, int] | None, float | None]:
    """Look up ``(color, opacity)`` from :data:`_NAME_STYLE_RULES` by name substring.

    Returns ``(None, None)`` when no rule matches — caller should then fall back
    to the renderer-wide defaults supplied at construction time.
    """
    lowered = name.lower()
    for needle, rgb, opacity in _NAME_STYLE_RULES:
        if needle in lowered:
            return rgb, opacity
    return None, None


def _try_import_trimesh():  # type: ignore[no-untyped-def]
    """Return the ``trimesh`` module or ``None`` if it isn't installed.

    ``trimesh`` is a core dep in ``pyproject.toml`` so this should always succeed on a
    proper install; guarding anyway lets the viewer stay usable in stripped-down sandboxes.
    """
    try:
        import trimesh  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    return trimesh


def _load_mesh_arrays(  # noqa: PLR0911 -- clean early-return chain is clearer than merging.
    mesh_path: str, scale: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load + centre + scale a triangle mesh, returning ``(vertices, faces)`` or ``None``.

    Returns ``None`` when: ``mesh_path`` is empty, the file doesn't exist, ``trimesh``
    can't parse it, or the loaded asset has no triangular faces. Any of these cases
    makes the renderer fall back to the cuboid placeholder. Results are memoised in
    :data:`_MESH_CACHE` keyed on (resolved absolute path, scale rounded to 6 decimals).

    The loaded vertices are **centred on their raw geometric centroid** before
    applying ``scale`` so the resulting mesh's local frame origin is its own
    centroid.  This pairs with :class:`SceneObject` adapters (currently only
    :mod:`hhtools.io.datasets.omomo`) which fold the matching
    ``scale · R · raw_centroid`` offset into ``positions`` — together the two
    halves preserve the upstream ``world_vert = obj_scale · (R · raw_vert) + obj_trans``
    formula bit-for-bit while giving ``SceneObject.positions`` an unambiguous
    "object centre in world" meaning.  Meshes whose raw centroid is already at
    the origin (most authoring-tool exports) get a no-op centring step, so this
    is safe for adapters that haven't been audited against the new contract.
    """
    if not mesh_path:
        return None
    resolved = Path(mesh_path).resolve()
    key = (str(resolved), round(float(scale), 6))
    cached = _MESH_CACHE.get(key)
    if cached is not None:
        return cached
    if not resolved.is_file():
        return None
    trimesh = _try_import_trimesh()
    if trimesh is None:
        return None
    try:
        loaded = trimesh.load(resolved, force="mesh", process=False)
    except Exception:
        return None
    verts = np.asarray(getattr(loaded, "vertices", np.zeros((0, 3))), dtype=np.float32)
    faces = np.asarray(getattr(loaded, "faces", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        return None
    centroid = verts.mean(axis=0).astype(np.float32)
    centred_scaled = ((verts - centroid) * float(scale)).astype(np.float32)
    scaled = centred_scaled, faces
    _MESH_CACHE[key] = scaled
    return scaled


class ObjectsRenderer:
    """Draw a list of :class:`SceneObject` trajectories into a Viser scene."""

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def] (viser.ViserServer)
        root_name: str = "/objects",
        color: tuple[int, int, int] = (255, 184, 108),
        opacity: float = 0.75,
    ) -> None:
        self._server = server
        self._root = root_name
        self._color = color
        self._opacity = float(opacity)
        self._objects: list[SceneObject] = []
        self._handles: list[object] = []
        self._visible = True
        self._version = 0  # suffix appended to handle names, bumped on every set_objects.

    # ------------------------------------------------------------------ public API

    def clear(self) -> None:
        """Drop every handle. Hide before remove — same rationale as the skeleton renderer:
        async Viser queues can otherwise flash the previous clip's geometry on the new clip
        for a frame or two when switching motions quickly.
        """
        for h in self._handles:
            try:
                h.visible = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                h.remove()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._handles = []
        self._objects = []

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for h in self._handles:
            try:
                h.visible = self._visible  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_objects(self, objects: Sequence[SceneObject]) -> None:
        """Switch to a new list of objects, creating either a mesh or a box handle each."""
        self.clear()
        self._objects = list(objects)
        self._version += 1
        for obj in self._objects:
            handle = self._create_handle(obj)
            self._handles.append(handle)
        # Caller is expected to call ``set_frame`` right after this so the handles' pose
        # reflects the user's current playback cursor, not the identity defaults above.

    def set_frame(self, frame: int) -> None:
        if not self._objects:
            return
        for obj, handle in zip(self._objects, self._handles, strict=False):
            idx = int(np.clip(frame, 0, obj.num_frames - 1))
            pos = obj.positions[idx]
            q_xyzw = obj.quaternions[idx]
            wxyz = (
                float(q_xyzw[3]),
                float(q_xyzw[0]),
                float(q_xyzw[1]),
                float(q_xyzw[2]),
            )
            try:
                handle.position = (float(pos[0]), float(pos[1]), float(pos[2]))  # type: ignore[attr-defined]
                handle.wxyz = wxyz  # type: ignore[attr-defined]
            except Exception:
                pass

    # ------------------------------------------------------------------ internals

    def _create_handle(self, obj: SceneObject):  # type: ignore[no-untyped-def]
        """Create a mesh handle if a usable mesh is available, else a cuboid placeholder.

        Style precedence (highest → lowest):

        1. Explicit per-object overrides: ``SceneObject.opacity`` / ``color`` when
           set by the adapter (used for one-off custom props).
        2. Name-based rules: :data:`_NAME_STYLE_RULES` maps semantic names (``terrain``,
           ``ground``, …) to a slate-gray opaque look so climbing geometry reads as
           solid ground rather than a translucent prop. Because these hints aren't
           persisted to the NPZ, this is what ensures a reloaded bundle renders the
           same as the source-tree version.
        3. Renderer-wide defaults: the ``color`` / ``opacity`` passed to ``__init__``,
           i.e. the generic prop style.
        """
        node_name = f"{self._root}/{_sanitize(obj.name)}_v{self._version}"
        name_rgb, name_opacity = _style_from_name(obj.name)
        opacity = float(obj.opacity) if obj.opacity is not None else (
            name_opacity if name_opacity is not None else self._opacity
        )
        color = obj.color if obj.color is not None else (
            name_rgb if name_rgb is not None else self._color
        )
        mesh = _load_mesh_arrays(obj.mesh_path, obj.scale) if obj.mesh_path else None
        if mesh is not None:
            verts, faces = mesh
            return self._server.scene.add_mesh_simple(
                name=node_name,
                vertices=verts,
                faces=faces,
                color=color,
                opacity=opacity,
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                visible=self._visible,
            )
        dims = tuple(float(d) for d in obj.extents.tolist())
        return self._server.scene.add_box(
            name=node_name,
            color=color,
            dimensions=dims,
            opacity=opacity,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=self._visible,
        )


def _sanitize(name: str) -> str:
    """Viser handle names share the scene-node namespace with slashes and reserved chars.
    We strip anything other than alnum / underscore / hyphen so user-provided object
    names like ``floor lamp`` don't produce invalid scene-node paths.
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "object"


__all__ = ["ObjectsRenderer"]
