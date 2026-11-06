"""Report subject means and percentage-point deltas against the paper."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = {
    ("eeg", "intra-subject"): (80.2, 96.9),
    ("eeg", "inter-subject"): (20.0, 46.2),
    ("meg", "intra-subject"): (37.7, 66.8),
    ("meg", "inter-subject"): (5.1, 17.6),
}

def collect_results(root: Path, seed: int) -> dict:
    grouped = {key: {} for key in PAPER}
    for path in sorted(root.rglob("test_results.json")):
        metadata_path = path.with_name("run_meta.json")
        if not metadata_path.is_file():
            print(f"Skipped without run_meta.json: {path}", file=sys.stderr)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("seed", -1)) != seed:
            continue
        key = (str(metadata.get("dataset", "")).lower(), str(metadata.get("exp_setting", "")))
        if key not in grouped:
            raise ValueError(f"Unrecognized protocol in {metadata_path}: {key}")
        subject = str(metadata.get("subject", ""))
        expected = {f"sub-{index:02d}" for index in range(1, 11 if key[0] == "eeg" else 5)}
        if subject not in expected:
            raise ValueError(f"Unrecognized subject in {metadata_path}: {subject}")
        if subject in grouped[key]:
            raise ValueError(f"Duplicate protocol/subject/seed: {key}, {subject}, {seed}. Choose one results root.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if len(payload) != 1:
                raise ValueError(f"Expected one test-loader result in {path}")
            payload = payload[0]
        top1, top5 = float(payload["test_top1_acc"]), float(payload["test_top5_acc"])
        if not (math.isfinite(top1) and math.isfinite(top5) and 0 <= top1 <= top5 <= 1):
            raise ValueError(f"Expected accuracy fractions in [0, 1] with Top-1 <= Top-5: {path}")
        grouped[key][subject] = {"top1": 100 * top1, "top5": 100 * top5, "path": str(path)}
    return grouped

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--json", type=Path, default=None, help="also save a machine-readable report")
    args = parser.parse_args()
    if not args.results.is_dir():
        print(f"No results directory: {args.results}")
    grouped = collect_results(args.results, args.seed)
    rows = []
    for key, reference in PAPER.items():
        subjects = grouped[key]
        expected_count = 10 if key[0] == "eeg" else 4
        row = {"dataset": key[0], "protocol": key[1], "seed": args.seed,
               "subjects": subjects, "expected_subjects": expected_count,
               "paper_top1": reference[0], "paper_top5": reference[1]}
        if not subjects:
            row["status"] = "not run"
        else:
            row["top1"] = sum(value["top1"] for value in subjects.values()) / len(subjects)
            row["top5"] = sum(value["top5"] for value in subjects.values()) / len(subjects)
            row["delta_top1_pp"] = row["top1"] - reference[0]
            row["delta_top5_pp"] = row["top5"] - reference[1]
            row["status"] = "complete subject coverage" if len(subjects) == expected_count else "partial subject coverage"
        rows.append(row)
        label = f"{key[0].upper()} {key[1]}"
        if subjects:
            print(f"{label}: {row['top1']:.2f}/{row['top5']:.2f}%, "
                  f"delta {row['delta_top1_pp']:+.2f}/{row['delta_top5_pp']:+.2f} pp; "
                  f"subjects {len(subjects)}/{expected_count}; {row['status']}")
        else:
            print(f"{label}: not run; paper {reference[0]:.1f}/{reference[1]:.1f}%")
    print("Single-seed means are descriptive comparisons with the paper's five-run means. "
          "Partial subject coverage cannot establish agreement with the overall paper mean.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
