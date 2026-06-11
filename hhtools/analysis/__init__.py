# SPDX-License-Identifier: Apache-2.0
"""Dataset visualization & analysis (``hhtools.analysis``).

A data-centric toolkit that turns a heterogeneous motion library (human source
clips and/or retargeted robot trajectories) into per-clip **metrics**, rule-based
**tags**, a semantic **embedding** and a diversity-aware **subset** selection.

The design mirrors the LIMMT / GQS framework (physics feasibility, behavioural
diversity, dynamic complexity) but stays *non-destructive*: nothing is ever
discarded automatically.  "Unreasonable" clips are merely tagged (``quality_bad``)
and the user decides whether to filter or keep them.

Pipeline (see :func:`hhtools.analysis.clip.analyze_clip`)::

    Motion | RobotCSV
        -> canonical.CanonicalMotionFeatures   (cross-source projection)
        -> metrics.compute_l0_metrics           (L0 skeleton kinematics + S_phy)
        -> scene_metrics.compute_l1/l2          (object / terrain aware)
        -> tags.assign_tags                      (rule labels)
        -> embedding.EmbeddingBackend.encode     (handcrafted A / PAE B)

The :mod:`hhtools.analysis.cluster` and :mod:`hhtools.analysis.subset` modules
operate on the *collection* of embeddings (2-D scatter, Global Weighted FPS).
"""

from __future__ import annotations

from hhtools.analysis.canonical import CanonicalMotionFeatures, project_to_canonical
from hhtools.analysis.clip import AnalyzableClip, analyze_clip

__all__ = [
    "AnalyzableClip",
    "CanonicalMotionFeatures",
    "analyze_clip",
    "project_to_canonical",
]
