"""Per-frame local transforms for a given Skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.math import transform as T
from hhtools.core.skeleton import Skeleton


@dataclass
class AnimationBuffer:
    """Time-sampled local joint transforms for a given skeleton.

    Attributes:
        skeleton: Character skeleton.
        local_transforms: ``(num_frames, num_bones, 7)`` array of ``[tx, ty, tz, qx, qy, qz, qw]``
            transforms relative to each bone's parent.
        framerate: Sample rate in frames per second.
    """

    skeleton: Skeleton
    local_transforms: NDArray
    framerate: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.local_transforms = np.asarray(self.local_transforms, dtype=np.float32)
        if self.local_transforms.ndim != 3 or self.local_transforms.shape[1:] != (
            self.skeleton.num_bones,
            7,
        ):
            raise ValueError(
                f"local_transforms shape {self.local_transforms.shape} does not match "
                f"(num_frames, {self.skeleton.num_bones}, 7)"
            )
        if self.framerate <= 0:
            raise ValueError(f"framerate must be positive; got {self.framerate}")

    # ----------------------------------------------------------------- queries

    @property
    def num_frames(self) -> int:
        return self.local_transforms.shape[0]

    @property
    def num_bones(self) -> int:
        return self.skeleton.num_bones

    @property
    def duration(self) -> float:
        """Total duration in seconds. Empty buffers return ``0.0``."""
        if self.num_frames == 0:
            return 0.0
        return (self.num_frames - 1) / self.framerate

    @property
    def delta_time(self) -> float:
        return 1.0 / self.framerate

    # ----------------------------------------------------------------- accessors

    def frame(self, index: int) -> NDArray:
        """Return the local transforms at ``index`` with shape ``(num_bones, 7)``."""
        if not 0 <= index < self.num_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self.num_frames})"
            )
        return self.local_transforms[index]

    def sample(self, time: float) -> NDArray:
        """Linearly interpolate (translation) and SLERP (rotation) to sample at ``time`` seconds."""
        if self.num_frames == 0:
            raise ValueError("Animation buffer is empty")
        if self.num_frames == 1:
            return self.local_transforms[0].copy()

        frame_f = float(np.clip(time * self.framerate, 0.0, self.num_frames - 1))
        i0 = int(np.floor(frame_f))
        i1 = min(i0 + 1, self.num_frames - 1)
        blend = float(frame_f - i0)

        if blend < 1e-6 or i0 == i1:
            return self.local_transforms[i0].copy()

        a = self.local_transforms[i0]
        b = self.local_transforms[i1]
        t_lerp = (1 - blend) * a[..., 0:3] + blend * b[..., 0:3]
        q_slerp = Q.slerp(a[..., 3:7], b[..., 3:7], blend)
        return np.concatenate([t_lerp, q_slerp], axis=-1)

    def compute_global_transforms(self, root_transforms: NDArray | None = None) -> NDArray:
        """Forward kinematics across all frames.

        Returns an ``(num_frames, num_bones, 7)`` array.
        """
        if root_transforms is None:
            root_transforms = T.identity(self.num_frames)
        elif root_transforms.shape != (self.num_frames, 7):
            raise ValueError(
                f"root_transforms shape {root_transforms.shape} must be "
                f"({self.num_frames}, 7)"
            )
        return self.skeleton.compute_global_transforms(self.local_transforms, root_transforms)
