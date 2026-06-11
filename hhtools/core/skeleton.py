"""Skeleton: a Hierarchy plus a reference (rest) local transform per bone.

The local rest transform defines each bone's position and orientation in its parent's frame when
the character is in its canonical pose (T-pose for humans, zero-pose for robots). Retargeting
algorithms use it as a reference for rest-pose alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.math import transform as T


@dataclass
class Skeleton:
    """Skeleton = Hierarchy + reference local transforms.

    Attributes:
        hierarchy: Bone topology.
        reference_local_transforms: ``(num_bones, 7)`` array of ``[tx, ty, tz, qx, qy, qz, qw]``
            describing each bone's local rest pose relative to its parent.
        up_axis: Character's up axis, one of ``"X" | "Y" | "Z"``. Stored as a hint for downstream
            retargeting and renderers.
        forward_axis: Character's forward axis, same convention.
    """

    hierarchy: Hierarchy
    reference_local_transforms: NDArray
    up_axis: str = "Z"
    forward_axis: str = "X"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.hierarchy.num_bones
        self.reference_local_transforms = np.asarray(
            self.reference_local_transforms, dtype=np.float32
        )
        if self.reference_local_transforms.shape != (n, 7):
            raise ValueError(
                f"reference_local_transforms shape {self.reference_local_transforms.shape} "
                f"must be ({n}, 7)"
            )

    # ----------------------------------------------------------------- helpers

    @property
    def num_bones(self) -> int:
        return self.hierarchy.num_bones

    @property
    def bone_names(self) -> list[str]:
        return self.hierarchy.bone_names

    def reference_positions(self) -> NDArray:
        """Global rest positions of every bone (shape ``(num_bones, 3)``)."""
        return self.compute_global_transforms(self.reference_local_transforms)[..., 0:3]

    # ----------------------------------------------------------------- kinematics

    def compute_global_transforms(
        self, local_transforms: NDArray, root_transform: NDArray | None = None
    ) -> NDArray:
        """Forward kinematics: local transforms -> global transforms.

        ``local_transforms`` has shape ``(..., num_bones, 7)``; the output has the same shape. The
        optional ``root_transform`` has shape ``(..., 7)`` and is composed with every root bone.
        """
        local_transforms = np.asarray(local_transforms, dtype=np.float32)
        if local_transforms.shape[-2:] != (self.num_bones, 7):
            raise ValueError(
                f"Expected local_transforms shape (..., {self.num_bones}, 7); got "
                f"{local_transforms.shape}"
            )
        leading = local_transforms.shape[:-2]
        if root_transform is None:
            root_transform = T.identity(*leading)
        else:
            root_transform = np.asarray(root_transform, dtype=np.float32)

        globals_ = np.empty_like(local_transforms)
        parents = self.hierarchy.parent_indices
        for i in range(self.num_bones):
            p = int(parents[i])
            if p == -1:
                globals_[..., i, :] = T.compose(root_transform, local_transforms[..., i, :])
            else:
                globals_[..., i, :] = T.compose(
                    globals_[..., p, :], local_transforms[..., i, :]
                )
        return globals_

    def compute_local_transforms(
        self, global_transforms: NDArray, root_transform: NDArray | None = None
    ) -> NDArray:
        """Inverse FK: global transforms -> local transforms (relative to parent)."""
        global_transforms = np.asarray(global_transforms, dtype=np.float32)
        if global_transforms.shape[-2:] != (self.num_bones, 7):
            raise ValueError(
                f"Expected global_transforms shape (..., {self.num_bones}, 7); got "
                f"{global_transforms.shape}"
            )
        leading = global_transforms.shape[:-2]
        if root_transform is None:
            root_transform = T.identity(*leading)
        else:
            root_transform = np.asarray(root_transform, dtype=np.float32)

        locals_ = np.empty_like(global_transforms)
        parents = self.hierarchy.parent_indices
        for i in range(self.num_bones):
            p = int(parents[i])
            if p == -1:
                parent_g = root_transform
            else:
                parent_g = global_transforms[..., p, :]
            locals_[..., i, :] = T.compose(T.inverse(parent_g), global_transforms[..., i, :])
        return locals_

    # ----------------------------------------------------------------- factories

    @classmethod
    def from_global_positions_and_quats(
        cls,
        hierarchy: Hierarchy,
        global_positions: NDArray,
        global_quats: NDArray | None = None,
        up_axis: str = "Z",
        forward_axis: str = "X",
    ) -> Skeleton:
        """Convenience factory: build a skeleton from global rest positions.

        When ``global_quats`` is omitted every bone's orientation is set to identity.
        """
        n = hierarchy.num_bones
        p = np.asarray(global_positions, dtype=np.float32)
        if p.shape != (n, 3):
            raise ValueError(f"global_positions shape {p.shape} must be ({n}, 3)")
        q = (
            np.asarray(global_quats, dtype=np.float32)
            if global_quats is not None
            else Q.identity(n)
        )
        if q.shape != (n, 4):
            raise ValueError(f"global_quats shape {q.shape} must be ({n}, 4)")

        locals_ = np.zeros((n, 7), dtype=np.float32)
        locals_[:, 6] = 1.0
        for i in range(n):
            parent = hierarchy.parent(i)
            if parent == -1:
                locals_[i, 0:3] = p[i]
                locals_[i, 3:7] = q[i]
            else:
                parent_q_inv = Q.conjugate(q[parent : parent + 1])
                locals_[i, 0:3] = Q.rotate(parent_q_inv, (p[i] - p[parent])[None])[0]
                locals_[i, 3:7] = Q.multiply(parent_q_inv, q[i : i + 1])[0]
        return cls(hierarchy, locals_, up_axis=up_axis, forward_axis=forward_axis)
