"""Render a per-frame skinned body mesh into a Viser scene.

This renderer pairs a static :class:`~hhtools.core.skinning.SkinnedMesh` with a live
:class:`~hhtools.core.motion.Motion` and, for every ``set_frame(i)`` call, does linear-blend
skinning on the rest vertices and pushes the deformed positions to a single
``add_mesh_simple`` handle.

Design choices:

* **One handle per renderer, per motion switch.** We keep exactly one mesh in the scene
  and rewrite its vertex buffer every frame.  Triangle topology never changes once a
  motion is set, so only ``vertices`` is rewritten — this is the Viser-friendly update
  path (``handle.vertices = ...`` sends just the new positions to the client).

* **Handle name carries a version suffix.** Just like the skeleton renderer, we version
  the handle's ``/skinned/mesh_v{N}`` name so an async remove-then-add on motion switch
  never lets the previous motion's topology stick around with the new motion's vertex
  counts — the Viser client would otherwise crash or render garbage on a mismatch.

* **Normals are auto-computed by Viser.**  We skip the LBS-on-normals pass for now.
  That would double per-frame cost, and Viser re-computes face normals from the updated
  vertex positions anyway.  Smooth shading from authored per-vertex normals can be
  revisited when we get a user-visible case for it (SMPL typically uses flat-per-face
  rendering which is exactly what we already produce).

* **Two attachment flavours.** The renderer accepts either
  :class:`~hhtools.core.skinning.SkinnedMesh` (rest-pose geometry + LBS weights, used for
  authored GLB / FBX rigs and deformed on the fly) *or*
  :class:`~hhtools.core.skinning.BakedMesh` (pre-computed per-frame vertex caches, used
  for SMPL / SMPL-H / SMPL-X where we want the exact output of the SMPL forward pass
  including corrective pose blendshapes).  Both produce the same Viser handle layout.

* **No-op without data.** A Motion that carries neither ``meta["skinned_mesh"]`` nor
  ``meta["baked_mesh"]`` simply leaves the handle cleared and ``set_frame`` short-
  circuits.  This keeps caller code in :mod:`hhtools.viewer.app` simple — it can always
  call ``set_motion`` / ``set_frame`` without pre-checking whether a mesh exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.core.motion import Motion
from hhtools.core.skinning import BakedMesh, SkinnedMesh, lbs_deform


@dataclass
class _State:
    motion: Motion
    mesh: SkinnedMesh | BakedMesh
    faces: NDArray  # (F, 3) int32
    is_baked: bool


class SkinnedMeshRenderer:
    """Viser scene node that draws the current frame's skinned body mesh.

    Typical wiring (see :mod:`hhtools.viewer.app`):

    .. code-block:: python

        renderer = SkinnedMeshRenderer(server, color=(180, 200, 220))
        renderer.set_motion(motion)  # no-op when motion has no attached SkinnedMesh
        # per-frame:
        renderer.set_frame(frame_idx)

    Toggle visibility with :meth:`set_visible` (which just flips the handle's
    ``visible`` flag; the underlying state stays alive so re-enabling is instant).
    """

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def]  (viser.ViserServer)
        root_name: str = "/skinned_mesh",
        color: tuple[int, int, int] = (180, 200, 220),
        opacity: float = 1.0,
    ) -> None:
        self._server = server
        self._root = root_name
        self._color = color
        self._opacity = float(opacity)
        self._state: _State | None = None
        self._handle = None
        self._visible = True
        self._version = 0

    # ------------------------------------------------------------------ public API

    def has_mesh(self) -> bool:
        """True after ``set_motion`` saw a motion carrying a SkinnedMesh attachment."""
        return self._state is not None

    def clear(self) -> None:
        """Drop the mesh handle. Safe to call repeatedly.

        Ordering (hide then remove) matches the skeleton renderer so no async vertex
        update from the old motion can flash into the scene after the new motion is
        wired up.
        """
        if self._handle is not None:
            try:
                self._handle.visible = False
            except Exception:
                pass
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None
        self._state = None

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            try:
                self._handle.visible = self._visible
            except Exception:
                pass

    def set_motion(self, motion: Motion) -> None:
        """Switch to a new motion; extracts the attached SkinnedMesh or BakedMesh if present.

        We check ``meta["skinned_mesh"]`` first (runtime LBS on GLB / FBX rigs), then fall
        back to ``meta["baked_mesh"]`` (pre-deformed SMPL / SMPL-H / SMPL-X vertex caches).
        When neither is present we still :meth:`clear` so a previously-loaded clip's mesh
        disappears — otherwise toggling between a GLB and a skeleton-only clip would leave
        the GLB mesh floating around over the new skeleton.
        """
        self.clear()
        meta = motion.meta if isinstance(motion.meta, dict) else {}
        skinned = meta.get("skinned_mesh")
        baked = meta.get("baked_mesh")

        if isinstance(skinned, SkinnedMesh):
            if skinned.num_joints != motion.num_bones:
                motion.meta.setdefault("viewer_warnings", []).append(
                    f"SkinnedMeshRenderer: mesh joint count {skinned.num_joints} "
                    f"!= motion bone count {motion.num_bones}; mesh disabled."
                )
                return
            self._state = _State(
                motion=motion,
                mesh=skinned,
                faces=skinned.triangles.astype(np.int32),
                is_baked=False,
            )
            self._version += 1
            return

        if isinstance(baked, BakedMesh):
            if baked.num_frames != motion.num_frames:
                motion.meta.setdefault("viewer_warnings", []).append(
                    f"SkinnedMeshRenderer: baked mesh frames {baked.num_frames} "
                    f"!= motion frames {motion.num_frames}; mesh disabled."
                )
                return
            self._state = _State(
                motion=motion,
                mesh=baked,
                faces=baked.triangles.astype(np.int32),
                is_baked=True,
            )
            self._version += 1

    def set_frame(self, frame: int) -> None:
        """Update / create the mesh handle in-place with the deformed vertices for ``frame``.

        For :class:`SkinnedMesh` we run LBS on the Motion's joint pose; for
        :class:`BakedMesh` we just index into the pre-computed ``vertices`` cache.
        """
        state = self._state
        if state is None:
            return
        frame = int(np.clip(frame, 0, state.motion.num_frames - 1))
        if state.is_baked:
            verts = state.mesh.frame(frame).astype(np.float32, copy=False)  # type: ignore[union-attr]
        else:
            verts = lbs_deform(
                state.mesh,  # type: ignore[arg-type]
                state.motion.positions[frame],
                state.motion.quaternions[frame],
            ).astype(np.float32, copy=False)

        if self._handle is None:
            self._handle = self._server.scene.add_mesh_simple(
                name=f"{self._root}/mesh_v{self._version}",
                vertices=verts,
                faces=state.faces,
                color=self._color,
                flat_shading=False,
                opacity=self._opacity,
                side="double",
            )
            self._handle.visible = self._visible
        else:
            self._handle.vertices = verts


__all__ = ["SkinnedMeshRenderer"]
