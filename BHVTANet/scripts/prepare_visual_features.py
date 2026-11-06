"""Download vision backbones and prepare reproducible feature caches for one YAML."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bhvtanet.data.common import ensure_mctf_multilevel_cache, ensure_visual_feature_caches
from verify_setup import load_config, mctf_settings


def image_paths(data_dir: Path, mode: str) -> list[str]:
    paths: list[str] = []
    for file_path in sorted(data_dir.glob(f"sub-*/{mode}.pt")):
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        paths.extend(str(value) for value in np.asarray(payload["img"], dtype=object).reshape(-1))
    if not paths:
        raise FileNotFoundError(f"No subject {mode}.pt files found below {data_dir}")
    return sorted(set(paths))


def prepare_anchor_cache(config: dict, mode: str, paths: list[str]) -> None:
    adapter = dict(config.get("target_adapter") or {})
    if not adapter.get("enabled", False):
        return
    anchor_specs = adapter.get("anchors") or [adapter]
    for spec in anchor_specs:
        anchor_dir = spec.get("anchor_feature_dir") or spec.get("feature_dir")
        anchor_model = spec.get("anchor_model_type") or spec.get("model_type")
        anchor_kind = str(spec.get("anchor_feature_kind", spec.get("feature_kind", "clean")))
        if not anchor_dir or not anchor_model:
            raise ValueError("An enabled target_adapter requires anchor_feature_dir and anchor_model_type.")
        anchor_config = copy.deepcopy(config)
        anchor_config["data"]["feature_dir"] = str(anchor_dir)
        anchor_config["data"]["model_type"] = str(anchor_model)
        anchor_config["ablation"]["primary_feature"] = anchor_kind
        anchor_config["ablation"]["secondary_feature"] = anchor_kind
        anchor_config["ablation"]["use_feature_interpolation"] = False
        anchor_config["mctf"] = {"enabled": False}
        ensure_visual_feature_caches(anchor_config, mode, paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing RN50, multi-level MCTF, and EEG anchor feature caches from raw images."
    )
    parser.add_argument("--config", required=True, type=Path, help="one of configs/*.yaml")
    parser.add_argument("--data-root", default=None, help="root containing things-eeg and things-meg")
    parser.add_argument("--anchor-root", default=None, help="directory for EEG anchor feature folders")
    parser.add_argument("--feature-subdir", default=None, help="visual cache folder name below Image_feature")
    parser.add_argument("--modes", default="train,test", help="comma-separated splits to prepare")
    parser.add_argument("--levels", default=None, help="optional comma-separated MCTF layers; default uses the YAML")
    args = parser.parse_args()

    config = load_config(args.config, args.data_root, args.anchor_root, args.feature_subdir)
    if args.levels:
        config["mctf"] = copy.deepcopy(mctf_settings(config))
        config["mctf"].setdefault("auto_extract", {})["levels"] = [
            token.strip() for token in args.levels.split(",") if token.strip()
        ]

    data_dir = Path(config["data"]["data_dir"])
    for mode in [token.strip() for token in args.modes.split(",") if token.strip()]:
        paths = image_paths(data_dir, mode)
        print(f"Preparing {mode} features for {len(paths)} unique images.")
        ensure_visual_feature_caches(config, mode, paths)
        ensure_mctf_multilevel_cache(config, mode, paths)
        if str(config.get("dataset", "")).lower() == "eeg":
            prepare_anchor_cache(config, mode, paths)
    print("Visual feature preparation finished.")


if __name__ == "__main__":
    main()
