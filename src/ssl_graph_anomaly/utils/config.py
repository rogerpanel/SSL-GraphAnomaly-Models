"""YAML configuration loading with a single level of ``defaults:`` inheritance."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(base))
    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent = cfg.pop("defaults", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def merge_configs(*configs: Mapping[str, Any]) -> dict:
    out: dict = {}
    for c in configs:
        out = _deep_merge(out, c)
    return out
