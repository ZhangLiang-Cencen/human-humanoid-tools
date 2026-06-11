# SPDX-License-Identifier: Apache-2.0
"""UI metadata catalog (tags, metrics, stages) for dataset analysis."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def _catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "configs" / "analysis" / "catalog.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    path = _catalog_path()
    if not path.is_file():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def tag_info(tag: str) -> dict[str, Any]:
    return load_catalog().get("tags", {}).get(tag, {"title": tag, "desc": ""})


def metric_info(key: str) -> dict[str, Any]:
    return load_catalog().get("metrics", {}).get(key, {"title": key, "desc": ""})


def category_info(key: str) -> dict[str, Any]:
    return load_catalog().get("categories", {}).get(key, {"title": key, "desc": ""})


__all__ = ["category_info", "load_catalog", "metric_info", "tag_info"]
