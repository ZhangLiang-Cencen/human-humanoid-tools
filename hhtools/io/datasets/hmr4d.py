"""Adapter for GVHMR / KungFuAthlete-style ``hmr4d_results.pt`` files.

Both the GVHMR release and the KungFuAthlete sample produced by the HMR4D pipeline share the
same on-disk layout -- a PyTorch checkpoint whose top-level dict contains
``smpl_params_global`` with ``body_pose`` (T, 63), ``betas`` (T, 10), ``global_orient`` (T, 3)
and ``transl`` (T, 3) tensors, plus some auxiliary network outputs.

We treat them as SMPL (not SMPL-H) because the pose dimensionality matches (21 body joints)
and no hand parameters are emitted by HMR4D.  However HMR4D stores a 21-joint body_pose which
matches SMPL-H convention, so we actually use SMPL-H if its weights are available, otherwise
we zero-pad to SMPL's 23-joint body pose.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.bodymodels.params import SmplMotionParams
from hhtools.core.motion import Motion
from hhtools.io.datasets._engine_cache import engine_for_params
from hhtools.io.datasets.base import DatasetAdapter, register_dataset

_DEFAULT_FRAMERATE = 30.0


def _to_numpy(x: Any) -> np.ndarray:
    try:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    except Exception:
        return np.asarray(x)


def _load_hmr4d(path: Path) -> SmplMotionParams:
    import torch  # noqa: F401 -- ensure torch is imported before unpickling

    from hhtools.bodymodels.compat import patch_chumpy_compat

    patch_chumpy_compat()
    try:
        import chumpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{path}: GVHMR/SMPL-H 需要 chumpy 才能读取 SMPL-H 权重。"
            "请执行 `uv pip install chumpy`（或 `uv sync --extra smpl`），"
            "或在 configs/body_models/smplh 下提供 SMPL-H neutral .npz 权重。"
        ) from exc

    data = __import__("torch").load(str(path), map_location="cpu", weights_only=False)
    block = data.get("smpl_params_global")
    if block is None:
        raise ValueError(
            f"{path} does not contain 'smpl_params_global'; is this an HMR4D results file?"
        )
    body_pose_21 = _to_numpy(block["body_pose"]).astype(np.float32)  # (T, 63)
    betas = _to_numpy(block["betas"]).astype(np.float32)
    global_orient = _to_numpy(block["global_orient"]).astype(np.float32)
    transl = _to_numpy(block["transl"]).astype(np.float32)

    if body_pose_21.ndim != 2 or body_pose_21.shape[1] != 63:
        raise ValueError(
            f"Unexpected body_pose shape {body_pose_21.shape} in {path}; expected (T, 63)."
        )
    # HMR4D fixes betas per frame; flatten to a single per-sequence shape vector for SMPL.
    if betas.ndim == 2:
        betas_flat = betas.mean(axis=0)
    else:
        betas_flat = betas.reshape(-1)

    # HMR4D body_pose has 21 body joints which matches SMPL-H layout; SMPL expects 23 joints
    # (69 dims) so we pad with zeros when using SMPL. We ship SMPL-H parameters by default
    # because SMPL-H weights are available in most user installations, but fall back to SMPL
    # when SMPL-H is absent (engine constructor will raise a clear error that the caller can
    # catch and retry with SMPL).
    return SmplMotionParams(
        surface_model="smplh",
        root_orient=global_orient,
        body_pose=body_pose_21,  # (T, 63) for SMPL-H
        betas=betas_flat,
        trans=transl,
        gender="neutral",
        framerate=_DEFAULT_FRAMERATE,
        hand_pose_left=None,
        hand_pose_right=None,
        up_axis="Y",  # HMR4D typically emits Y-up
        meta={"dataset": "hmr4d", "source_path": str(path)},
    )


class _Hmr4dBase(DatasetAdapter):
    requires = "smplh"
    file_patterns = ("*.pt", "*.pth")

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.pt")):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"HMR4D results file not found: {p}")
        return p

    def load_params(self, sequence_id: str) -> SmplMotionParams:
        return _load_hmr4d(self._resolve(sequence_id))

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        with_mesh = bool(kwargs.pop("with_mesh", False))
        progress_callback = kwargs.pop("progress_callback", None)
        params = self.load_params(sequence_id)
        try:
            engine = engine_for_params(params)
        except FileNotFoundError:
            # SMPL-H weights missing; fall back to SMPL (pad to 69 body-pose dims).
            padded = np.zeros((params.num_frames, 69), dtype=np.float32)
            padded[:, :63] = params.body_pose
            params = SmplMotionParams(
                surface_model="smpl",
                root_orient=params.root_orient,
                body_pose=padded,
                betas=params.betas,
                trans=params.trans,
                gender=params.gender,
                framerate=params.framerate,
                up_axis=params.up_axis,
                meta=params.meta,
            )
            engine = engine_for_params(params)
        return engine.to_motion(
            params,
            name=Path(sequence_id).stem,
            source_format=f"hmr4d/{params.surface_model}",
            return_mesh=with_mesh,
            progress_callback=progress_callback,
        )


@register_dataset
class GvhmrAdapter(_Hmr4dBase):
    name = "gvhmr"
    display_name = "GVHMR (World-grounded Video HMR)"


@register_dataset
class KungFuAthleteAdapter(_Hmr4dBase):
    name = "kungfu_athlete"
    display_name = "KungFuAthlete (HMR4D extraction)"


__all__ = ["GvhmrAdapter", "KungFuAthleteAdapter"]
