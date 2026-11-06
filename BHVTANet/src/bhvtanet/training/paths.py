"""Portable resource paths shared by training and setup checks."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def register_path_resolvers():
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "data_root",
        lambda: os.environ.get("BHVTANET_DATA_ROOT", str(PROJECT_ROOT / "data")),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "anchor_root",
        lambda: os.environ.get("BHVTANET_ANCHOR_ROOT", str(PROJECT_ROOT / "data" / "anchors")),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "feature_subdir",
        lambda: os.environ.get("BHVTANET_FEATURE_SUBDIR", "BHVTANet"),
        replace=True,
    )


def resolve_project_paths(config):
    """Resolve relative resource and output locations against the checkout."""
    from omegaconf import OmegaConf

    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    path_fields = [
        (config, "save_dir"),
        (config.get("data", {}), "data_dir"),
        (config.get("data", {}), "image_root"),
        (config.get("data", {}), "feature_dir"),
        (config.get("target_adapter", {}), "anchor_feature_dir"),
    ]
    path_fields.append((config.get("mctf", {}).get("auto_extract", {}), "image_root"))
    for anchor in config.get("target_adapter", {}).get("anchors", []) or []:
        path_fields.extend((anchor, key) for key in ("anchor_feature_dir", "feature_dir"))
    for container, key in path_fields:
        value = container.get(key)
        if value and not Path(str(value)).is_absolute():
            container[key] = str((PROJECT_ROOT / str(value)).resolve())
    mctf_config = config.get("mctf", {})
    if mctf_config.get("level_feature_paths"):
        mctf_config["level_feature_paths"] = [
            str(Path(str(path))) if Path(str(path)).is_absolute()
            else str((PROJECT_ROOT / str(path)).resolve())
            for path in mctf_config["level_feature_paths"]
        ]
    return config
