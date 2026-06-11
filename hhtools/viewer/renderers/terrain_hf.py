"""Render a :class:`TerrainHeightfield` into a Viser scene.

Triangulates the heightfield's regular grid into ``2 * (nx-1) * (ny-1)``
triangles and pushes the resulting mesh via
:meth:`viser.SceneApi.add_mesh_simple`.  The same heightfield array drives:

* this on-screen visualisation,
* the MuJoCo ``<hfield>`` collision asset compiled by
  :func:`hhtools.retarget.interaction_mesh.collision.build_collision_model_with_hfield`,
* and the PARC-format ``.pkl`` shipped to training,

so what the user sees is exactly what the optimizer feels and what the
training rig will be exposed to.

A static terrain is the typical case (no per-frame update needed); the
:meth:`set_frame` method is a no-op kept only so the renderer can be
swapped in beside :class:`ObjectsRenderer` without callers having to
special-case it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.scene import TerrainHeightfield


class TerrainHeightfieldRenderer:
    """Draw a heightfield as a triangulated mesh in a Viser scene.

    Parameters
    ----------
    server
        Live :class:`viser.ViserServer` instance.
    root_name
        Scene-graph node name; new mesh handles are versioned with a
        ``_v{N}`` suffix to avoid colliding with the previous clip's
        terrain when the user switches motions.
    color
        Default RGB triplet (0–255) for cells; overridden per-call by
        :meth:`set_terrain` if the caller passes a colour.
    opacity
        Default mesh opacity (0–1).
    """

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def] (viser.ViserServer)
        root_name: str = "/terrain_hf",
        color: tuple[int, int, int] = (100, 116, 139),  # slate-500
        opacity: float = 1.0,
    ) -> None:
        self._server = server
        self._root = root_name
        self._color = color
        self._opacity = float(opacity)
        self._terrain: TerrainHeightfield | None = None
        self._handle: object | None = None
        self._visible = True
        self._version = 0

    # ------------------------------------------------------------------ public API

    def clear(self) -> None:
        """Drop the current handle.  Hide before remove for the same async
        reason :class:`ObjectsRenderer` does it."""
        if self._handle is not None:
            try:
                self._handle.visible = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                self._handle.remove()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._handle = None
        self._terrain = None

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            try:
                self._handle.visible = self._visible  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_terrain(
        self,
        terrain: TerrainHeightfield | None,
        *,
        color: tuple[int, int, int] | None = None,
        opacity: float | None = None,
    ) -> None:
        """Switch to a new heightfield (or clear when ``None``)."""
        self.clear()
        if terrain is None:
            return

        verts, faces = terrain.triangulate()
        if int(verts.shape[0]) < 3 or int(faces.shape[0]) < 1:
            return

        self._terrain = terrain
        self._version += 1
        node = f"{self._root}_v{self._version}"

        c = color if color is not None else self._color
        o = float(opacity) if opacity is not None else self._opacity
        self._handle = self._server.scene.add_mesh_simple(
            name=node,
            vertices=np.asarray(verts, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            color=c,
            opacity=o,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=self._visible,
        )

    def set_frame(self, frame: int) -> None:  # noqa: ARG002 — interface parity
        """Static heightfields have no per-frame update."""
        return

    # ------------------------------------------------------------------ helpers

    @property
    def terrain(self) -> TerrainHeightfield | None:
        return self._terrain


__all__ = ["TerrainHeightfieldRenderer"]
