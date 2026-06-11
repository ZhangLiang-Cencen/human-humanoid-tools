"""Discovery utilities for locally installed SMPL / SMPL-H / SMPL-X weights.

Search order (first hit wins):

1. ``HHTOOLS_BODY_MODELS`` environment variable (explicit user override).
2. ``<cwd>/configs/body_models/``         -- project-local convention used by hhtools tutorials.
3. ``<repo root>/configs/body_models/``   -- convenience for users who launch commands from
   sub-directories of a cloned repository.
4. ``~/.cache/hhtools/body_models/``      -- platform-aware fallback (honours
   :envvar:`HHTOOLS_CACHE_DIR`).

Within each root we expect the smplx-library layout::

    <root>/smpl/SMPL_<GENDER>.{pkl,npz}
    <root>/smplh/SMPLH_<GENDER>.{pkl,npz}
    <root>/smplx/SMPLX_<GENDER>.{pkl,npz}
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir

from hhtools.utils.paths import HHTOOLS_CACHE_ENV

HHTOOLS_BODY_MODELS_ENV = "HHTOOLS_BODY_MODELS"

_FAMILY_DIRS = {"smpl": "smpl", "smplh": "smplh", "smplx": "smplx"}
_GENDER_CASES = {"neutral": "NEUTRAL", "male": "MALE", "female": "FEMALE"}


def _platform_cache_root() -> Path:
    override = os.environ.get(HHTOOLS_CACHE_ENV)
    base = Path(override) if override else Path(user_cache_dir("hhtools", "hhtools"))
    return base / "body_models"


def _repo_roots() -> list[Path]:
    """Walk upwards from ``cwd`` looking for marker files that identify a project root."""
    seen: list[Path] = []
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if parent in seen:
            break
        seen.append(parent)
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return [parent]
    return []


def body_model_search_paths() -> list[Path]:
    """Return the ordered list of directories that may contain body model weights.

    If :envvar:`HHTOOLS_BODY_MODELS` is set, its value is treated as the *only* search root,
    which lets advanced users and tests pin the lookup to a specific directory.
    Otherwise the chain above is consulted in order.  The returned list always contains at
    least one entry so callers can safely use the first element as a default.
    """
    override = os.environ.get(HHTOOLS_BODY_MODELS_ENV)
    if override:
        return [Path(override)]

    paths: list[Path] = []
    cwd_configs = Path.cwd() / "configs" / "body_models"
    paths.append(cwd_configs)

    for repo in _repo_roots():
        candidate = repo / "configs" / "body_models"
        if candidate not in paths:
            paths.append(candidate)

    paths.append(_platform_cache_root())
    return paths


def default_body_model_root() -> Path:
    """Return the first candidate root used for downloads / registration tips.

    When the user has ``configs/body_models`` available in the project we prefer that.
    """
    for path in body_model_search_paths():
        if path.is_dir():
            return path
    return _platform_cache_root()


def find_body_model(
    family: str,
    gender: str = "neutral",
    *,
    roots: list[Path] | None = None,
) -> Path | None:
    """Locate a weight file for *family* / *gender* in one of the search roots.

    Returns ``None`` if no matching file was found.  Accepts both ``.npz`` and ``.pkl``
    extensions; ``.npz`` is preferred when both are present because it does not require the
    ``chumpy`` compatibility shim.
    """
    family = family.lower()
    gender = gender.lower()
    if family not in _FAMILY_DIRS:
        raise ValueError(f"Unknown body model family: {family!r}")
    if gender not in _GENDER_CASES:
        raise ValueError(f"Unknown gender: {gender!r}")

    token = _GENDER_CASES[gender]
    dir_name = _FAMILY_DIRS[family]
    filename_prefix = family.upper()

    candidate_roots = roots if roots is not None else body_model_search_paths()
    for root in candidate_roots:
        family_dir = root / dir_name
        if not family_dir.is_dir():
            continue
        for ext in (".npz", ".pkl"):
            candidate = family_dir / f"{filename_prefix}_{token}{ext}"
            if candidate.is_file():
                return candidate
    return None


def check_body_models(root: Path | str | None = None) -> dict[str, bool]:
    """Return ``{family: bool}`` for whether a weight file is found for the given root.

    When *root* is ``None`` the full :func:`body_model_search_paths` chain is consulted.
    """
    roots: list[Path] | None
    roots = [Path(root)] if root is not None else None
    return {
        family: find_body_model(family, "neutral", roots=roots) is not None
        or find_body_model(family, "male", roots=roots) is not None
        or find_body_model(family, "female", roots=roots) is not None
        for family in _FAMILY_DIRS
    }


__all__ = [
    "HHTOOLS_BODY_MODELS_ENV",
    "body_model_search_paths",
    "check_body_models",
    "default_body_model_root",
    "find_body_model",
]
