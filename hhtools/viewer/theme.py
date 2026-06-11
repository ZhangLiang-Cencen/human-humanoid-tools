"""Viser theme + colour palette for hhtools.

Light studio theme: pale canvas, readable typography, and saturated accents so
controls stay visible.  Renderer colours are chosen to read on a bright ground
plane (see the viewer's ``add_grid`` ``plane_opacity``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    # Canvas & chrome (light)
    background: str = "#f4f6fb"
    panel: str = "#ffffff"
    grid: str = "#94a3b8"  # slate-400 — grid lines on white

    # Interactive accent
    accent: str = "#0284c7"  # sky-600
    accent_dim: str = "#0369a1"

    # Semantic
    good: str = "#0d9488"
    warn: str = "#d97706"
    error: str = "#dc2626"

    # Body renderers (distinct on light ground)
    human: str = "#0369a1"  # deep cyan — skeleton
    human_scaled: str = "#6d28d9"  # violet-700 — scaled overlay
    robot: str = "#b45309"  # amber-700 — robot / capsules
    joint_bead: str = "#1e293b"  # slate-800

    # Scene geometry hints (labels / markdown only; grid uses explicit RGB in app)
    terrain: str = "#64748b"
    prop: str = "#ea580c"

    # Typography
    text: str = "#0f172a"
    text_muted: str = "#475569"

    # Sidebar markdown / status (muted so they don't compete with the 3D view)
    ui_ok: str = "#0f766e"       # teal-700
    ui_warn: str = "#b45309"    # amber-700
    ui_error: str = "#b91c1c"   # red-700
    ui_info: str = "#0369a1"    # sky-700
    ui_muted_line: str = "#64748b"  # slate-500

    # Scene grid (light ground) — softer than high-contrast charcoal
    grid_cell: str = "#cbd5e1"      # slate-300
    grid_section: str = "#94a3b8"  # slate-400


PALETTE = Palette()

TITLEBAR = "hhtools · Human ↔ Humanoid Studio"

__all__ = ["PALETTE", "Palette", "TITLEBAR"]
