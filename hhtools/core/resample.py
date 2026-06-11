"""Framerate resampling for Motion / AnimationBuffer.

Translation is interpolated linearly; rotation with SLERP to preserve geodesic smoothness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q


def _resample_translation(values: NDArray, src_fps: float, dst_fps: float, num_dst: int) -> NDArray:
    """Linear interpolation of translations along the time axis."""
    src_n = values.shape[0]
    src_times = np.arange(src_n, dtype=np.float64) / src_fps
    dst_times = np.arange(num_dst, dtype=np.float64) / dst_fps
    # Clamp to the source time range to avoid extrapolation.
    dst_times = np.clip(dst_times, src_times[0], src_times[-1])

    # Interpolate each channel independently (translations have 3 components).
    out_shape = (num_dst, *values.shape[1:])
    out = np.empty(out_shape, dtype=values.dtype)
    flat = values.reshape(src_n, -1)
    interp = np.empty((num_dst, flat.shape[1]), dtype=values.dtype)
    for c in range(flat.shape[1]):
        interp[:, c] = np.interp(dst_times, src_times, flat[:, c])
    out = interp.reshape(out_shape)
    return out


def _resample_quaternions(
    quats: NDArray, src_fps: float, dst_fps: float, num_dst: int
) -> NDArray:
    """SLERP over time for an ``(num_frames, ..., 4)`` quaternion stack."""
    src_n = quats.shape[0]
    src_times = np.arange(src_n, dtype=np.float64) / src_fps
    dst_times = np.arange(num_dst, dtype=np.float64) / dst_fps
    dst_times = np.clip(dst_times, src_times[0], src_times[-1])

    out = np.empty((num_dst, *quats.shape[1:]), dtype=quats.dtype)
    for i, t in enumerate(dst_times):
        idx = np.searchsorted(src_times, t, side="right") - 1
        idx = int(np.clip(idx, 0, src_n - 2))
        t0 = src_times[idx]
        t1 = src_times[idx + 1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        out[i] = Q.slerp(quats[idx], quats[idx + 1], float(alpha))
    return out


def resample_time_series(values: NDArray, src_fps: float, dst_fps: float) -> NDArray:
    """Linearly resample a time series along axis 0."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("Framerates must be positive")
    src_n = values.shape[0]
    if src_n < 2:
        return values.copy()
    duration = (src_n - 1) / src_fps
    num_dst = int(np.floor(duration * dst_fps)) + 1
    return _resample_translation(values, src_fps, dst_fps, num_dst)


def resample_quaternion_series(values: NDArray, src_fps: float, dst_fps: float) -> NDArray:
    """SLERP-resample a time series of quaternions along axis 0."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("Framerates must be positive")
    src_n = values.shape[0]
    if src_n < 2:
        return values.copy()
    duration = (src_n - 1) / src_fps
    num_dst = int(np.floor(duration * dst_fps)) + 1
    return _resample_quaternions(values, src_fps, dst_fps, num_dst)


def resample_motion(motion, target_fps: float):  # type: ignore[no-untyped-def]
    """Resample a :class:`hhtools.core.motion.Motion` to ``target_fps``.

    This is a lightweight wrapper to keep ``Motion`` free of the resampling implementation.
    """
    from hhtools.core.motion import Motion  # local import to avoid a cycle

    if not isinstance(motion, Motion):
        raise TypeError(f"Expected Motion, got {type(motion)}")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if abs(target_fps - motion.framerate) < 1e-6:
        return motion

    pos = resample_time_series(motion.positions, motion.framerate, target_fps)
    quat = resample_quaternion_series(motion.quaternions, motion.framerate, target_fps)

    # Heightfield terrain is time-invariant — it survives resampling
    # untouched.  SceneObject trajectories are dropped here as they
    # were before this refactor; resampling object trajectories needs
    # its own treatment that is out of scope for this helper.
    return Motion(
        name=motion.name,
        hierarchy=motion.hierarchy,
        positions=pos,
        quaternions=quat,
        framerate=target_fps,
        up_axis=motion.up_axis,
        source_format=motion.source_format,
        meta={**motion.meta, "resampled_from_fps": motion.framerate},
        terrain=motion.terrain,
    )


def resample_motion_with_objects(motion, target_fps: float):  # type: ignore[no-untyped-def]
    """Like :func:`resample_motion` but also resamples :attr:`Motion.objects` trajectories.

    Static :attr:`Motion.terrain` is preserved unchanged.  Each :class:`~hhtools.core.scene.SceneObject`
    gets linearly resampled positions and SLERP-resampled quaternions to the new frame count.
    """
    from dataclasses import replace

    from hhtools.core.motion import Motion
    from hhtools.core.scene import SceneObject

    if not isinstance(motion, Motion):
        raise TypeError(f"Expected Motion, got {type(motion)}")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if abs(target_fps - motion.framerate) < 1e-6:
        return motion

    pos = resample_time_series(motion.positions, motion.framerate, target_fps)
    quat = resample_quaternion_series(motion.quaternions, motion.framerate, target_fps)

    new_meta = {**motion.meta, "resampled_from_fps": motion.framerate}

    new_objects: list[SceneObject] = []
    for ob in motion.objects or []:
        op = resample_time_series(ob.positions, motion.framerate, target_fps)
        oq = resample_quaternion_series(ob.quaternions, motion.framerate, target_fps)
        new_objects.append(
            replace(
                ob,
                positions=op.astype(np.float32, copy=False),
                quaternions=oq.astype(np.float32, copy=False),
            )
        )

    return Motion(
        name=motion.name,
        hierarchy=motion.hierarchy,
        positions=pos,
        quaternions=quat,
        framerate=target_fps,
        up_axis=motion.up_axis,
        source_format=motion.source_format,
        meta=new_meta,
        objects=new_objects,
        terrain=motion.terrain,
    )
