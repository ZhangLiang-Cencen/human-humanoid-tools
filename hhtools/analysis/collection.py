# SPDX-License-Identifier: Apache-2.0
"""Collection-level orchestration: analyze many clips, embed, cluster, summarise.

This is the entry point the web layer calls.  It runs the per-clip pipeline
(:func:`hhtools.analysis.clip.analyze_clip`), then the collection-level steps that
need the whole distribution: embedding fit, 2-D scatter, clustering,
distribution-relative tags, and a histogram / tag-count summary for the UI.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hhtools.analysis import cluster as _cluster
from hhtools.analysis import tags as _tags
from hhtools.analysis.clip import AnalyzableClip, analyze_clip
from hhtools.analysis.config import load_config
from hhtools.analysis.embedding import make_embedding

ProgressCb = Callable[[float, str], None]

# Metric keys surfaced as histograms / scatter axes in the UI.
_NUMERIC_METRIC_KEYS: tuple[str, ...] = (
    "duration_s",
    "complexity",
    "joint_kinetic_energy",
    "joint_accel_energy",
    "root_speed_xy",
    "root_speed_z",
    "root_turn_rate",
    "com_height_range",
    "airborne_ratio",
    "path_efficiency",
    "step_freq",
    "leg_energy",
    "arm_energy",
    "inverted_ratio",
    "max_torso_tilt_deg",
    "s_phy",
)


def analyze_entries(
    entries: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
    embedding_name: str | None = None,
    progress: ProgressCb | None = None,
) -> list[AnalyzableClip]:
    """Analyze a list of ``{clip_id, source_path, dataset, folder_label}`` dicts."""
    cfg = cfg or load_config()
    embedding_name = embedding_name or cfg.get("embedding", {}).get("backend", "handcrafted")

    clips: list[AnalyzableClip] = []
    total = max(len(entries), 1)
    for i, e in enumerate(entries):
        if progress is not None:
            progress(0.05 + 0.7 * (i / total), f"分析 {e.get('clip_id', '')}")
        clips.append(
            analyze_clip(
                e["source_path"],
                clip_id=e.get("clip_id") or str(i),
                source_path=e["source_path"],
                dataset=e.get("dataset", ""),
                folder_label=e.get("folder_label", ""),
                cfg=cfg,
            )
        )

    ok = [c for c in clips if c.error is None and c.metrics]
    if progress is not None:
        progress(0.8, "计算 embedding 与聚类…")

    if ok:
        backend = make_embedding(embedding_name, cfg)
        try:
            vecs = backend.fit_encode(ok)
            emb = np.stack(vecs, axis=0)
            scatter = _cluster.project_2d(emb)
            labels = _cluster.cluster(emb)
            for c, v, xy, lab in zip(ok, vecs, scatter, labels):
                c.embedding = [round(float(x), 5) for x in v.tolist()]
                c.scatter = (round(float(xy[0]), 5), round(float(xy[1]), 5))
                c.cluster_id = int(lab)
        except Exception:  # noqa: BLE001 - embedding optional; metrics still usable
            pass

        _tags.assign_dataset_tags(ok, cfg)

    if progress is not None:
        progress(0.95, "汇总分布…")
    return clips


def build_summary(clips: list[AnalyzableClip], cfg: dict[str, Any]) -> dict[str, Any]:
    """Histograms per numeric metric + tag counts + cluster counts."""
    ok = [c for c in clips if c.error is None and c.metrics]
    n = len(ok)

    histograms: dict[str, Any] = {}
    for key in _NUMERIC_METRIC_KEYS:
        vals = np.array(
            [float(c.metrics.get(key)) for c in ok if c.metrics.get(key) is not None],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        nbins = int(min(30, max(5, round(np.sqrt(vals.size)))))
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        counts, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
        histograms[key] = {
            "counts": counts.astype(int).tolist(),
            "edges": [round(float(e), 5) for e in edges.tolist()],
            "min": round(lo, 5),
            "max": round(hi, 5),
            "mean": round(float(vals.mean()), 5),
            "median": round(float(np.median(vals)), 5),
        }

    tag_counts: dict[str, int] = {}
    for c in ok:
        for t in c.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    cluster_counts: dict[str, int] = {}
    for c in ok:
        if c.cluster_id is not None:
            cluster_counts[str(c.cluster_id)] = cluster_counts.get(str(c.cluster_id), 0) + 1

    folder_counts: dict[str, int] = {}
    for c in ok:
        folder_counts[c.folder_label] = folder_counts.get(c.folder_label, 0) + 1

    return {
        "num_clips": len(clips),
        "num_ok": n,
        "num_error": len(clips) - n,
        "numeric_keys": list(histograms.keys()),
        "histograms": histograms,
        "tag_counts": tag_counts,
        "tag_order": _tags.all_known_tags(),
        "cluster_counts": cluster_counts,
        "folder_counts": folder_counts,
    }


__all__ = ["analyze_entries", "build_summary"]
