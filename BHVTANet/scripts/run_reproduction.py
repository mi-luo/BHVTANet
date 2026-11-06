"""Run selected protocols once with one seed on Windows or Linux."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from verify_setup import CONFIGS, DEFAULTS, ROOT

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", nargs="+", choices=(*CONFIGS, "all"), default=["eeg_intra", "meg_intra"])
    parser.add_argument("--subjects", default="all", help="all or a comma-separated subject list")
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--anchor-root", default=None, help="directory containing EEG anchor feature folders")
    parser.add_argument("--feature-subdir", default=None, help="visual cache folder name below Image_feature")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print commands without checking data or training")
    args = parser.parse_args()
    for field in ("data_root", "anchor_root", "output_root"):
        value = getattr(args, field)
        if value:
            setattr(args, field, str(Path(value).expanduser().resolve()))
    child_env = os.environ.copy()
    child_env["PYTHONNOUSERSITE"] = "1"
    protocols = list(CONFIGS) if "all" in args.protocols else list(dict.fromkeys(args.protocols))
    for protocol in protocols:
        dataset = protocol.split("_", 1)[0]
        setting, epochs, lr, brain = DEFAULTS[protocol]
        config = ROOT / "configs" / f"{protocol}.yaml"
        command = [sys.executable, "-s", str(ROOT / "main.py"), "--config", str(config),
                   "--dataset", dataset, "--exp_setting", setting, "--subjects", args.subjects,
                   "--seed", str(args.seed), "--epoch", str(epochs), "--lr", str(lr),
                   "--brain_backbone", brain, "--vision_backbone", "RN50", "--gpu", args.gpu]
        check = [sys.executable, "-s", str(ROOT / "scripts" / "verify_setup.py"), "--config", str(config),
                 "--subjects", args.subjects]
        if args.data_root:
            command.extend(["--data-root", args.data_root])
            check.extend(["--data-root", args.data_root])
        if args.anchor_root:
            command.extend(["--anchor-root", args.anchor_root])
            check.extend(["--anchor-root", args.anchor_root])
        if args.feature_subdir:
            command.extend(["--feature-subdir", args.feature_subdir])
            check.extend(["--feature-subdir", args.feature_subdir])
        if args.output_root:
            command.extend(["--output-root", args.output_root])
        print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(check, cwd=ROOT, env=child_env, check=True)
            subprocess.run(command, cwd=ROOT, env=child_env, check=True)

if __name__ == "__main__":
    main()
