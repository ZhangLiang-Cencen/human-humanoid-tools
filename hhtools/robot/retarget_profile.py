"""Per-robot retarget defaults from ``robot.yaml``'s ``retarget:`` block."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from hhtools.retarget.newton_basic.config import (
    FeetStabilizerConfig,
    ScalerConfig,
    load_scaler_config,
)

if TYPE_CHECKING:
    from hhtools.core.motion import Motion
    from hhtools.retarget.calibration.calibration import RobotRetargetCalibration
    from hhtools.robot.base import RobotPreset
    from hhtools.robot.loader import URDFRobotModel

# Matches :data:`hhtools.retarget.calibration.calibration._CANONICAL_HUMAN_HEIGHT_M`.
_DEFAULT_HUMAN_HEIGHT_BY_REFERENCE: dict[str, float] = {
    "smpl": 1.65,
    "smplx": 1.65,
    "gvhmr": 1.65,
    "soma_bvh": 1.65,
    "lafan_bvh": 1.65,
    "fbx": 1.65,
    "glb": 1.65,
}


def _retarget_block(preset: "RobotPreset") -> dict[str, Any]:
    block = preset.meta.get("retarget")
    return dict(block) if isinstance(block, dict) else {}


def _reference_block(preset: "RobotPreset", reference: str) -> dict[str, Any]:
    block = _retarget_block(preset)
    refs = block.get("references")
    if isinstance(refs, dict):
        ref_cfg = refs.get(reference)
        if isinstance(ref_cfg, dict):
            return dict(ref_cfg)
    return {}


def _workspace_robots_root() -> Path | None:
    """``configs/robots/`` in the source tree, if present."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "robots"
        if candidate.is_dir():
            return candidate
    return None


def _workspace_robot_dir(preset_name: str) -> Path | None:
    """``configs/robots/<name>/`` in the source tree, if present."""

    root = _workspace_robots_root()
    if root is None:
        return None
    candidate = root / preset_name
    return candidate if candidate.is_dir() else None


def _scaler_search_roots(preset: "RobotPreset") -> list[Path]:
    """Preset dir first, then same-named workspace bundle (user upload shadowing)."""

    roots: list[Path] = [preset.root_dir.resolve()]
    ws = _workspace_robot_dir(preset.name)
    if ws is not None:
        resolved = ws.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _scaler_rel_candidates(
    preset: "RobotPreset",
    reference: str,
) -> list[str]:
    """Scaler yaml filenames declared in ``robot.yaml`` for ``reference``."""

    rels: list[str] = []
    user_rel = _reference_block(preset, reference).get("scaler_config")
    if user_rel:
        rels.append(str(user_rel))

    ws = _workspace_robot_dir(preset.name)
    if ws is not None and ws.resolve() != preset.root_dir.resolve():
        yaml_path = ws / "robot.yaml"
        if yaml_path.is_file():
            try:
                with yaml_path.open("r", encoding="utf-8") as fp:
                    data = yaml.safe_load(fp) or {}
                refs = (data.get("retarget") or {}).get("references") or {}
                ref_cfg = refs.get(reference) or {}
                ws_rel = ref_cfg.get("scaler_config")
                if ws_rel and str(ws_rel) not in rels:
                    rels.append(str(ws_rel))
            except Exception:  # noqa: BLE001 — optional metadata
                pass
    return rels


def bundled_scaler_path(preset: "RobotPreset", reference: str) -> Path | None:
    """Return a preset-local scaler yaml when ``robot.yaml`` declares one.

    Scaler YAML is optional and must be referenced explicitly via
    ``retarget.references.<reference>.scaler_config``.  All robots otherwise
    derive scaler parameters from Web / CLI calibration
    (``retarget_calibration_<reference>.yaml``).
    """

    for root in _scaler_search_roots(preset):
        for rel in _scaler_rel_candidates(preset, reference):
            path = (root / rel).resolve()
            if path.is_file():
                return path
    return None


def default_human_height(
    preset: "RobotPreset",
    reference: str,
    *,
    fallback: float = 1.7,
) -> float:
    """Default source-human height when the request omits one.

    Prefer an optional per-robot bundled scaler's ``human_height_assumption``,
    else a reference-family canonical stature (1.65 m for SMPL / SOMA / LAFAN /
    FBX / GLB), else ``fallback``.
    """

    bundled = bundled_scaler_path(preset, reference)
    if bundled is not None:
        try:
            cfg = load_scaler_config(bundled)
        except Exception:  # noqa: BLE001 - fall back to a sane constant
            pass
        else:
            h = float(getattr(cfg, "human_height_assumption", 0.0) or 0.0)
            if h > 0.1:
                return h

    from hhtools.retarget.calibration.calibration import normalize_calibration_reference

    ref = normalize_calibration_reference(reference)
    if ref in _DEFAULT_HUMAN_HEIGHT_BY_REFERENCE:
        return _DEFAULT_HUMAN_HEIGHT_BY_REFERENCE[ref]
    return float(fallback)


def resolve_retarget_scaler_config(
    preset: "RobotPreset",
    reference: str,
    *,
    calibration: "RobotRetargetCalibration | None",
    model: "URDFRobotModel",
    motion: "Motion",
    human_height: float,
) -> ScalerConfig:
    """Prefer calibration-derived scaler; fall back to optional bundled yaml."""

    if calibration is not None and model is not None:
        from hhtools.retarget.calibration import build_scaler_config_from_calibration

        return build_scaler_config_from_calibration(
            calibration, model, motion, human_height=human_height,
        )

    bundled = bundled_scaler_path(preset, reference)
    if bundled is not None:
        cfg = load_scaler_config(bundled)
        if motion is not None:
            from hhtools.retarget.newton_basic.scaler import (
                adapt_scaler_config_for_hierarchy,
            )

            return adapt_scaler_config_for_hierarchy(cfg, motion.hierarchy)
        return cfg

    raise ValueError(
        f"robot {preset.name!r} has no bundled scaler for reference "
        f"{reference!r} and no calibration file"
    )


def build_feet_stabilizer_config(
    preset: "RobotPreset",
    reference: str,
) -> FeetStabilizerConfig | None:
    """Feet stabilizer knobs from ``retarget.feet`` / per-reference overrides."""

    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)
    feet_raw = ref_cfg.get("feet_stabilizer") or block.get("feet_stabilizer")
    if not isinstance(feet_raw, dict):
        return None
    return FeetStabilizerConfig(
        up_axis=str(feet_raw.get("up_axis", preset.up_axis)),  # type: ignore[arg-type]
        forward_axis=str(feet_raw.get("forward_axis", preset.forward_axis)),  # type: ignore[arg-type]
        ground_contact_z=float(feet_raw.get("ground_contact_z", 0.0)),
        min_foot_clearance=float(feet_raw.get("min_foot_clearance", 0.0)),
        max_ground_correction=float(feet_raw.get("max_ground_correction", 0.05)),
        ground_uprightness_range=float(feet_raw.get("ground_uprightness_range", 0.30)),
        foot_planting_velocity_threshold=float(
            feet_raw.get("foot_planting_velocity_threshold", 0.0)
        ),
        foot_planting_height_margin=float(
            feet_raw.get("foot_planting_height_margin", 0.02)
        ),
        foot_planting_release_frames=int(
            feet_raw.get("foot_planting_release_frames", 3)
        ),
        min_lateral_separation=float(feet_raw.get("min_lateral_separation", 0.0)),
        smoothing_max_rate=float(feet_raw.get("smoothing_max_rate", 0.008)),
        left_foot_name=str(feet_raw.get("left_foot_name", "left_ankle")),
        right_foot_name=str(feet_raw.get("right_foot_name", "right_ankle")),
        left_toe_name=feet_raw.get("left_toe_name"),
        right_toe_name=feet_raw.get("right_toe_name"),
        hips_name=str(feet_raw.get("hips_name", "hips")),
    )


def build_pipeline_config_for_preset(
    preset: "RobotPreset",
    reference: str,
    *,
    ik_iterations: int,
):
    """Merge ``retarget:`` defaults into :class:`PipelineConfig`."""

    from hhtools.retarget.newton_basic.pipeline import PipelineConfig

    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)

    def _pick(key: str, default: Any) -> Any:
        if key in ref_cfg:
            return ref_cfg[key]
        if key in block:
            return block[key]
        return default

    return PipelineConfig(
        ik_iterations=int(ik_iterations),
        joint_limit_weight=float(_pick("joint_limit_weight", 10.0)),
        smooth_joint_filter_weight=float(_pick("smooth_joint_filter_weight", 5.5)),
        num_initialization_frames=int(_pick("num_initialization_frames", 0)),
        num_stabilization_frames=int(_pick("num_stabilization_frames", 0)),
        apply_feet_stabilizer=bool(_pick("apply_feet_stabilizer", False)),
    )
