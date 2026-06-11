"""SMPL / SMPL-H / SMPL-X forward engine that produces :class:`hhtools.core.motion.Motion`.

Given a :class:`hhtools.bodymodels.params.SmplMotionParams` instance the engine:

1. Loads the requested MPI body model via the :mod:`smplx` package (lazily, so ``hhtools.core``
   users never pay the import cost).
2. Runs a batched forward pass to obtain joint positions and optionally vertices / faces.
3. Converts the axis-angle pose vector into per-joint local quaternions and composes them
   along the kinematic tree to obtain global quaternions.
4. Produces a :class:`Motion` with canonical joint names (see :mod:`hhtools.bodymodels.layout`).

The module honours the MPI non-commercial license by never downloading weights; users must
provide them via the :func:`hhtools.bodymodels.paths.find_body_model` search chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

ProgressCallback = Callable[[float, str], None]

import numpy as np

from hhtools.bodymodels.compat import patch_chumpy_compat
from hhtools.bodymodels.layout import BodyModelLayout, layout_for
from hhtools.bodymodels.params import SmplMotionParams
from hhtools.bodymodels.paths import find_body_model
from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion

Family = Literal["smpl", "smplh", "smplx"]
Gender = Literal["neutral", "male", "female"]


@dataclass
class ForwardResult:
    """Dense output of a body model forward pass."""

    joints: np.ndarray  # (T, J, 3) in the model's world frame
    quaternions_global: np.ndarray  # (T, J, 4) xyzw
    vertices: np.ndarray | None  # (T, V, 3) or None when not requested
    faces: np.ndarray | None  # (F, 3) or None when not requested
    layout: BodyModelLayout


class SmplxEngine:
    """Thin, cached wrapper around :class:`smplx.create` for hhtools.

    The underlying ``smplx`` model is constructed once per (family, gender, num_betas)
    combination and re-used across forward calls.  Forward passes run on CPU and in chunks
    to avoid large peak-memory spikes on long sequences.
    """

    def __init__(
        self,
        family: Family,
        gender: Gender = "neutral",
        *,
        num_betas: int = 10,
        model_path: Path | str | None = None,
        use_pca: bool = False,
        chunk_size: int = 128,
        model_root: Path | str | None = None,
    ) -> None:
        self.family = family
        self.gender = gender
        self.num_betas = num_betas
        self.use_pca = use_pca
        self.chunk_size = chunk_size
        self.layout = layout_for(family)

        # The chumpy weights shipped by MPI for SMPL(-H) cannot be unpickled without the
        # compatibility shims below, which monkey-patch the missing ``inspect`` / ``numpy``
        # symbols on Python 3.12+.  The shim is a no-op on older runtimes.
        patch_chumpy_compat()

        resolved = self._resolve_weights(model_path, model_root)
        self._model_path = resolved

        import smplx
        import torch

        self._torch = torch
        self._smplx = smplx
        # ``smplx.create`` expects either a weights-root directory containing family
        # sub-directories, or a file path. We always pass a file path for determinism.
        kwargs: dict[str, Any] = {
            "model_type": family,
            "gender": gender,
            "num_betas": num_betas,
            "batch_size": 1,
            "ext": resolved.suffix.lstrip("."),
        }
        if family in ("smplh", "smplx"):
            kwargs["use_pca"] = use_pca
            kwargs["flat_hand_mean"] = True
        self._model = smplx.create(str(resolved.parent.parent), **kwargs)
        self._model = self._model.eval()
        # Cache per-joint parent array for downstream FK.
        self._parents = np.asarray(self._model.parents.detach().cpu().numpy(), dtype=np.int64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        params: SmplMotionParams,
        *,
        return_mesh: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> ForwardResult:
        """Run SMPL forward and produce joint positions, quaternions and optional vertices."""
        if params.surface_model != self.family:
            raise ValueError(
                f"SmplxEngine was built for {self.family!r} but received params for "
                f"{params.surface_model!r}. Construct a new engine for this family."
            )

        torch = self._torch
        T = params.num_frames
        layout = self.layout
        J = layout.num_joints

        all_joints = np.empty((T, J, 3), dtype=np.float32)
        all_quats = np.empty((T, J, 4), dtype=np.float32)
        all_vertices: np.ndarray | None = None
        faces_arr: np.ndarray | None = None
        if return_mesh:
            faces_arr = np.asarray(self._model.faces, dtype=np.int32)

        parents = self._parents[:J]

        stage = "烘焙身体网格" if return_mesh else "计算骨架"
        if progress_callback is not None:
            progress_callback(0.0, f"{stage} 0/{T} 帧")
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            fn_kwargs = self._chunk_kwargs(params, start, end)
            with torch.no_grad():
                out = self._model(return_verts=return_mesh, **fn_kwargs)
            joints_chunk = out.joints.detach().cpu().numpy()[:, :J].astype(np.float32)
            all_joints[start:end] = joints_chunk
            # Local -> global quaternions.
            local_quats = self._local_quaternions(params, start, end)
            all_quats[start:end] = self._compose_global(local_quats, parents)
            if return_mesh:
                verts_chunk = out.vertices.detach().cpu().numpy().astype(np.float32)
                if all_vertices is None:
                    all_vertices = np.empty((T, verts_chunk.shape[1], 3), dtype=np.float32)
                all_vertices[start:end] = verts_chunk
            if progress_callback is not None:
                progress_callback(
                    end / max(T, 1),
                    f"{stage} {end}/{T} 帧",
                )

        return ForwardResult(
            joints=all_joints,
            quaternions_global=all_quats,
            vertices=all_vertices,
            faces=faces_arr,
            layout=layout,
        )

    def to_motion(
        self,
        params: SmplMotionParams,
        *,
        name: str = "smpl_motion",
        source_format: str | None = None,
        return_mesh: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Motion:
        """Run forward and wrap the result as a canonical :class:`Motion`.

        When ``return_mesh=True`` the SMPL forward is invoked with ``return_verts=True``
        and the resulting ``(T, V, 3)`` vertex cache is attached to ``motion.meta`` as a
        :class:`hhtools.core.skinning.BakedMesh`. The viewer's SkinnedMeshRenderer will
        then render this body alongside the skeleton, giving you the real SMPL skin
        instead of just capsules. Baking costs 1–3 s per 300-frame clip and adds
        ~25–40 MB to the returned Motion, so callers should only opt in when they
        intend to display the mesh.
        """
        # Mesh baking is best-effort — the skeleton is what the pipeline
        # actually needs, so a vertex-output failure (torch OOM, shape-dir
        # mismatch for an odd ``num_betas`` fit, smplx upstream bug, …)
        # should never prevent the Motion from loading.  We retry forward
        # without mesh on failure and flag the shortfall in meta.
        baked_mesh_error: str | None = None
        if progress_callback is not None:
            progress_callback(0.02, f"初始化 {self.family} 模型…")
        try:
            result = self.forward(
                params,
                return_mesh=return_mesh,
                progress_callback=progress_callback,
            )
        except Exception as err:  # noqa: BLE001 — mesh failure must not kill load
            if not return_mesh:
                raise
            baked_mesh_error = f"{type(err).__name__}: {err}"
            result = self.forward(params, return_mesh=False, progress_callback=progress_callback)

        hierarchy = Hierarchy.from_parent_indices(
            list(result.layout.joint_names),
            [int(p) for p in result.layout.parents_array().tolist()],
        )
        meta = dict(params.meta)
        meta.update(
            surface_model=self.family,
            gender=self.gender,
            num_betas=int(params.betas.reshape(-1).shape[0]),
            joint_layout=self.family,
            model_path=str(self._model_path),
        )
        if return_mesh and result.vertices is not None and result.faces is not None:
            from hhtools.core.skinning import BakedMesh  # local import keeps core lean

            meta["baked_mesh"] = BakedMesh(
                vertices=result.vertices,
                triangles=result.faces,
            )
        elif return_mesh:
            # Forward succeeded but didn't produce vertices (e.g. a model
            # family without mesh support or a retry after baked_mesh_error).
            meta["baked_mesh_unavailable"] = True
            if baked_mesh_error is not None:
                meta["baked_mesh_error"] = baked_mesh_error
        src = source_format if source_format is not None else f"{self.family}"
        return Motion(
            name=name,
            hierarchy=hierarchy,
            positions=result.joints,
            quaternions=result.quaternions_global,
            framerate=params.framerate,
            up_axis=params.up_axis,
            source_format=src,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_weights(
        self, model_path: Path | str | None, model_root: Path | str | None
    ) -> Path:
        if model_path is not None:
            p = Path(model_path)
            if not p.is_file():
                raise FileNotFoundError(f"Body model weights not found at {p}")
            return p
        roots = [Path(model_root)] if model_root is not None else None
        p = find_body_model(self.family, self.gender, roots=roots)
        if p is None:
            raise FileNotFoundError(
                f"No {self.family.upper()} ({self.gender}) weight file found. "
                "Run `hhtools bodymodel setup` for download instructions."
            )
        return p

    def _chunk_kwargs(self, params: SmplMotionParams, start: int, end: int) -> dict[str, Any]:
        torch = self._torch
        chunk = end - start

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, dtype=np.float32))

        betas = np.asarray(params.betas, dtype=np.float32)
        if betas.ndim == 1:
            betas_t = t(np.broadcast_to(betas, (chunk, betas.shape[0])).copy())
        else:
            betas_t = t(betas[start:end])

        kwargs: dict[str, Any] = {
            "betas": betas_t,
            "global_orient": t(params.root_orient[start:end]),
            "body_pose": t(params.body_pose[start:end]),
            "transl": t(params.trans[start:end]),
        }
        if self.family in ("smplh", "smplx"):
            # Zero-pad missing hand poses so the batch dim stays consistent (see SMPL-X
            # comment below). SMPL-H / SMPL-X use 45 axis-angle dims per hand when
            # ``use_pca=False`` (our default).
            hand_dim = 45
            lh = params.hand_pose_left[start:end] if params.hand_pose_left is not None else np.zeros(
                (chunk, hand_dim), dtype=np.float32
            )
            rh = params.hand_pose_right[start:end] if params.hand_pose_right is not None else np.zeros(
                (chunk, hand_dim), dtype=np.float32
            )
            kwargs["left_hand_pose"] = t(lh)
            kwargs["right_hand_pose"] = t(rh)
        if self.family == "smplx":
            # SMPL-X's ``torch.cat`` pipeline inside body_models.py requires every component
            # tensor to match the batch size of ``betas``. When the input params omit any of
            # jaw / leye / reye we must fill in zero-tensors with the correct chunk size or
            # smplx raises "Sizes of tensors must match except in dimension 1".
            jaw = params.jaw_pose[start:end] if params.jaw_pose is not None else np.zeros(
                (chunk, 3), dtype=np.float32
            )
            leye = params.leye_pose[start:end] if params.leye_pose is not None else np.zeros(
                (chunk, 3), dtype=np.float32
            )
            reye = params.reye_pose[start:end] if params.reye_pose is not None else np.zeros(
                (chunk, 3), dtype=np.float32
            )
            kwargs["jaw_pose"] = t(jaw)
            kwargs["leye_pose"] = t(leye)
            kwargs["reye_pose"] = t(reye)
            # SMPL-X always needs an expression tensor with matching batch size; default to
            # zeros when the dataset does not ship any facial blendshape coefficients.
            num_exp = int(getattr(self._model, "num_expression_coeffs", 10))
            if params.expression is not None:
                exp = np.asarray(params.expression, dtype=np.float32)
                if exp.ndim == 1:
                    exp = np.broadcast_to(exp, (chunk, exp.shape[0])).copy()
                else:
                    exp = exp[start:end]
                # Dataset may ship 50 FLAME expression coeffs (Motion-X); SMPL-X's
                # default is 10 so we truncate. Zero-pad the other direction just in case.
                if exp.shape[1] > num_exp:
                    exp = exp[:, :num_exp]
                elif exp.shape[1] < num_exp:
                    pad = np.zeros((chunk, num_exp - exp.shape[1]), dtype=np.float32)
                    exp = np.concatenate([exp, pad], axis=1)
            else:
                exp = np.zeros((chunk, num_exp), dtype=np.float32)
            kwargs["expression"] = t(exp)
        return kwargs

    def _local_quaternions(self, params: SmplMotionParams, start: int, end: int) -> np.ndarray:
        """Build an ``(F, J, 4)`` xyzw tensor of per-joint *local* quaternions for the chunk.

        Joints not represented in the body-model pose vector (e.g. SMPL's hand tips or the
        eyes of SMPL-X when ``*_eye_pose`` is absent) default to identity quaternions.
        """
        layout = self.layout
        chunk = end - start
        J = layout.num_joints
        locals_ = np.tile(Q.identity(), (chunk, J, 1)).astype(np.float32)

        locals_[:, 0] = Q.from_axis_angle(params.root_orient[start:end])

        body_pose = params.body_pose[start:end]
        # body_pose always covers joints 1 .. num_body_pose_joints (depending on family)
        if self.family == "smpl":
            # body_pose has 23 joints
            for i in range(23):
                locals_[:, i + 1] = Q.from_axis_angle(body_pose[:, i * 3 : (i + 1) * 3])
        else:
            # body_pose covers joints 1..21 (=21 joints = 63 dims)
            for i in range(21):
                locals_[:, i + 1] = Q.from_axis_angle(body_pose[:, i * 3 : (i + 1) * 3])

        if self.family == "smplh":
            # 22..36 = left hand (15 joints), 37..51 = right hand (15 joints)
            if params.hand_pose_left is not None:
                lh = params.hand_pose_left[start:end]
                for i in range(15):
                    locals_[:, 22 + i] = Q.from_axis_angle(lh[:, i * 3 : (i + 1) * 3])
            if params.hand_pose_right is not None:
                rh = params.hand_pose_right[start:end]
                for i in range(15):
                    locals_[:, 37 + i] = Q.from_axis_angle(rh[:, i * 3 : (i + 1) * 3])
        elif self.family == "smplx":
            # 22=jaw, 23=leye, 24=reye, 25..39 = left hand, 40..54 = right hand
            if params.jaw_pose is not None:
                locals_[:, 22] = Q.from_axis_angle(params.jaw_pose[start:end])
            if params.leye_pose is not None:
                locals_[:, 23] = Q.from_axis_angle(params.leye_pose[start:end])
            if params.reye_pose is not None:
                locals_[:, 24] = Q.from_axis_angle(params.reye_pose[start:end])
            if params.hand_pose_left is not None:
                lh = params.hand_pose_left[start:end]
                for i in range(15):
                    locals_[:, 25 + i] = Q.from_axis_angle(lh[:, i * 3 : (i + 1) * 3])
            if params.hand_pose_right is not None:
                rh = params.hand_pose_right[start:end]
                for i in range(15):
                    locals_[:, 40 + i] = Q.from_axis_angle(rh[:, i * 3 : (i + 1) * 3])
        return locals_

    @staticmethod
    def _compose_global(local_quats: np.ndarray, parents: np.ndarray) -> np.ndarray:
        """Compose per-joint local quaternions along the kinematic tree to global quaternions.

        Both input and output are ``(F, J, 4)`` xyzw. Root quaternion is passed through.
        """
        F, J, _ = local_quats.shape
        globals_ = np.empty_like(local_quats)
        globals_[:, 0] = local_quats[:, 0]
        for j in range(1, J):
            p = int(parents[j])
            if p < 0:
                globals_[:, j] = local_quats[:, j]
            else:
                globals_[:, j] = Q.multiply(globals_[:, p], local_quats[:, j])
        return globals_


# Legacy shim retained for backwards compatibility with M3 scaffold tests -- now forwards to
# the real engine and therefore raises a more informative error if weights are unavailable.
def forward(
    params: SmplMotionParams,
    *,
    gender: Gender = "neutral",
    model_path: Path | str | None = None,
    model_root: Path | str | None = None,
) -> ForwardResult:
    """Convenience wrapper that constructs a one-shot :class:`SmplxEngine` and runs forward."""
    engine = SmplxEngine(
        params.surface_model,
        gender=gender,
        model_path=model_path,
        model_root=model_root,
    )
    return engine.forward(params)


__all__ = ["ForwardResult", "SmplxEngine", "forward"]
