"""Interactive setup wizard for SMPL / SMPL-H / SMPL-X body model weights.

The MPI licensed weights cannot be redistributed, so this module intentionally does *not*
download anything automatically.  It only points users at the correct registration URLs and
verifies that they have placed the files at a location ``hhtools`` can find.
"""

from __future__ import annotations

from pathlib import Path

from hhtools.bodymodels.paths import (
    body_model_search_paths,
    check_body_models,
    default_body_model_root,
)

BODY_MODEL_URLS = {
    "smpl": "https://smpl.is.tue.mpg.de",
    "smplh": "https://mano.is.tue.mpg.de",
    "smplx": "https://smpl-x.is.tue.mpg.de",
}

# Accepted layout documented for each family. File names match the ``smplx`` package
# defaults so that weights downloaded from the MPI websites work out of the box.
BODY_MODEL_FILES = {
    "smpl": ("smpl/SMPL_NEUTRAL.pkl", "smpl/SMPL_MALE.pkl", "smpl/SMPL_FEMALE.pkl"),
    "smplh": ("smplh/SMPLH_MALE.pkl", "smplh/SMPLH_FEMALE.pkl"),
    "smplx": (
        "smplx/SMPLX_NEUTRAL.npz",
        "smplx/SMPLX_MALE.npz",
        "smplx/SMPLX_FEMALE.npz",
    ),
}


def run_wizard(root: Path | str | None = None) -> None:
    """Print step-by-step instructions for obtaining body model weights.

    This creates the ``smpl/``, ``smplh/`` and ``smplx/`` sub-directories at *root* (so the
    user has a clear target) but never touches the network.
    """
    target = Path(root) if root is not None else default_body_model_root()
    target.mkdir(parents=True, exist_ok=True)
    print("# hhtools body model setup")
    print("# =========================")
    print()
    print("This wizard prepares local directories for MPI's SMPL family weights.")
    print("Weights are released under a NON-COMMERCIAL research license; hhtools does NOT")
    print("bundle or auto-download them.  Please register on the official websites, accept")
    print("the license terms and download the files yourself.\n")
    print(f"Target directory: {target}\n")
    print("Search chain inspected by `hhtools bodymodel check`:")
    for idx, p in enumerate(body_model_search_paths(), start=1):
        marker = "<-- will install here" if p == target else ""
        print(f"  {idx}. {p}  {marker}")
    print()
    for family, url in BODY_MODEL_URLS.items():
        (target / family).mkdir(parents=True, exist_ok=True)
        print(f"[{family.upper()}]  Register + download at {url}")
        for relpath in BODY_MODEL_FILES[family]:
            print(f"  -> place at {target / relpath}")
        print()
    print("Run `hhtools bodymodel check` to verify the installation.\n")


__all__ = [
    "BODY_MODEL_FILES",
    "BODY_MODEL_URLS",
    "check_body_models",
    "default_body_model_root",
    "run_wizard",
]
