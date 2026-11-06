"""Check dependencies and local artifacts without loading models or starting training."""
from __future__ import annotations
import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIGS = ("eeg_intra", "eeg_inter", "meg_intra", "meg_inter")
DEFAULTS = {
    "eeg_intra": ("intra-subject", 80, 1e-4, "EEGProject"),
    "eeg_inter": ("inter-subject", 50, 1e-4, "EEGProject"),
    "meg_intra": ("intra-subject", 80, 1e-3, "EEGProject"),
    "meg_inter": ("inter-subject", 150, 1e-3, "EEGProject"),
}
DEPENDENCIES = (
    "torch", "torchvision", "pytorch_lightning", "omegaconf", "numpy", "pandas",
    "PIL", "cv2", "open_clip", "clip", "timm", "safetensors", "tensorboard",
    "tqdm", "einops", "scipy", "loguru", "ftfy", "regex",
)

def load_config(path: Path, data_root: str | None = None,
                anchor_root: str | None = None, feature_subdir: str | None = None) -> dict:
    from omegaconf import OmegaConf
    from bhvtanet.training.paths import register_path_resolvers, resolve_project_paths
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if data_root:
        os.environ["BHVTANET_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
    if anchor_root:
        os.environ["BHVTANET_ANCHOR_ROOT"] = str(Path(anchor_root).expanduser().resolve())
    if feature_subdir:
        os.environ["BHVTANET_FEATURE_SUBDIR"] = feature_subdir
    register_path_resolvers()
    setting, epochs, lr, brain = DEFAULTS.get(path.stem, ("intra-subject", 80, 1e-4, "EEGProject"))
    defaults = {"exp_setting": setting, "epoch": epochs, "lr": lr, "brain_backbone": brain,
                "vision_backbone": "RN50", "z_dim": 1024, "seed": 200}
    return OmegaConf.to_container(resolve_project_paths(OmegaConf.merge(defaults, OmegaConf.load(path))), resolve=True)

def subject_names(spec: str, dataset: str) -> list[str]:
    if spec == "all":
        return [f"sub-{index:02d}" for index in range(1, 11 if dataset == "eeg" else 5)]
    names = []
    for part in spec.split(","):
        part = part.strip()
        number = part.removeprefix("sub-")
        if not number.isdigit():
            raise ValueError(f"Invalid subject: {part}")
        name = f"sub-{int(number):02d}"
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one subject is required")
    allowed = subject_names("all", dataset)
    if any(name not in allowed for name in names):
        raise ValueError(f"Subjects for {dataset} must be among {allowed}")
    return names

def feature_name(model: str, kind: str, mode: str, config: dict) -> str:
    model = model.replace("-", "_")
    blur = config.get("Blur", {})
    suffix = {"clean": "clean",
              "semantic": f"semantic_fovea_k{blur.get('soft_k', 51)}_g{blur.get('soft_g', 3)}",
              "perceptual": f"perceptual_gaussian_k{blur.get('hard_k', 31)}"}
    if kind not in suffix:
        raise ValueError(f"Unsupported feature kind: {kind}")
    return f"{model}_{suffix[kind]}_{mode}.pt"

def mctf_settings(config: dict) -> dict:
    """Read MCTF settings without importing model dependencies."""
    return dict(config.get("mctf", {}) or {})

def required_artifacts(config: dict, subjects: str) -> list[Path]:
    dataset = str(config["dataset"]).lower()
    selected = subject_names(subjects, dataset)
    if config["exp_setting"] == "inter-subject":
        selected = subject_names("all", dataset)
    data = config["data"]
    artifacts = [Path(data["data_dir"]) / subject / f"{mode}.pt"
                 for subject in selected for mode in ("train", "test")]
    ablation = config.get("ablation", {})
    kinds = {ablation.get("primary_feature", "semantic")}
    if ablation.get("use_feature_interpolation", True):
        kinds.add(ablation.get("secondary_feature", "perceptual"))
    features = Path(data["feature_dir"])
    subdir = str(data.get("feature_subdir", ""))
    if subdir and features.name != subdir and (features / subdir).is_dir():
        features = features / subdir
    artifacts.extend(features / feature_name(str(data["model_type"]), kind, mode, config)
                     for kind in sorted(kinds) for mode in ("train", "test"))
    mctf = mctf_settings(config)
    if mctf.get("enabled", False) and mctf.get("feature_mode") == "multilevel_cache":
        artifacts.extend(Path(path) for path in mctf.get("level_feature_paths", []))
    adapter = config.get("target_adapter", {})
    if adapter.get("enabled", False):
        for spec in adapter.get("anchors") or [adapter]:
            directory = spec.get("anchor_feature_dir") or spec.get("feature_dir")
            model = spec.get("anchor_model_type") or spec.get("model_type")
            kind = spec.get("anchor_feature_kind", spec.get("feature_kind", "clean"))
            if not directory or not model:
                raise ValueError("Enabled target adapter requires its anchor directory and model")
            artifacts.extend(Path(directory) / feature_name(str(model), str(kind), mode, config)
                             for mode in ("train", "test"))
    return list(dict.fromkeys(artifacts))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", help="check one YAML; repeat for several")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--anchor-root", default=None, help="directory containing EEG anchor feature folders")
    parser.add_argument("--feature-subdir", default=None, help="visual cache folder name below Image_feature")
    parser.add_argument("--files-only", action="store_true", help="skip dependencies other than OmegaConf")
    args = parser.parse_args()
    if args.data_root:
        args.data_root = str(Path(args.data_root).expanduser().resolve())
    if args.anchor_root:
        args.anchor_root = str(Path(args.anchor_root).expanduser().resolve())
    os.chdir(ROOT)
    missing_dependencies = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    if not args.files_only:
        print("Missing dependencies: " + (", ".join(missing_dependencies) or "none"))
    if importlib.util.find_spec("omegaconf") is None:
        raise SystemExit("Install OmegaConf before checking YAML resources: python -m pip install omegaconf==2.3.0")
    configs = args.config or [ROOT / "configs" / f"{name}.yaml" for name in CONFIGS]
    failed = bool(missing_dependencies) and not args.files_only
    for path in configs:
        print(f"\n[{path.name}]")
        try:
            config = load_config(path, args.data_root, args.anchor_root, args.feature_subdir)
            artifacts = required_artifacts(config, args.subjects)
        except (KeyError, ValueError, OSError, TypeError) as error:
            print(f"CONFIG ERROR: {error}")
            failed = True
            continue
        missing = [path for path in artifacts if not path.is_file() or path.stat().st_size == 0]
        print(f"Required files present: {len(artifacts) - len(missing)}/{len(artifacts)}")
        for artifact in missing:
            print(f"MISSING: {artifact}")
        failed |= bool(missing)
        if config["exp_setting"] == "intra-subject" and config["data"].get("use_test_as_val", False):
            print("PROTOCOL: validation reuses test data; checkpoint selection is not independent of test data.")
        mctf = mctf_settings(config)
        if mctf.get("single_level_override") is not None:
            print(f"MODEL: fixed visual level {mctf['single_level_override']}; multi-depth aggregation is bypassed.")
    print("\nThis check verifies file presence and import availability, not tensor contents or numerical reproduction.")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
