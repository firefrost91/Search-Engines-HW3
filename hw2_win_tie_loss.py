#!/usr/bin/env python3
"""
Compute win/tie/loss counts using per-query MAP relative to a baseline.
Also prints overall MAP and per-query MAP differences.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict


QREL_PATH = Path("INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed")
TREC_EVAL = Path("INPUT_DIR/trec_eval-9.0.4")
DEFAULT_BASELINE_PARAM = Path("TEST_DIR/HW2-Exp-0.param")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--baseline", default=None,
                   help="Baseline run file (TREC format). If omitted, use baseline param outputPath.")
    p.add_argument("--baseline-param", default=str(DEFAULT_BASELINE_PARAM),
                   help="Baseline param file used to infer outputPath when --baseline is not set")
    p.add_argument("--runs", nargs="*",
                   help="Run files to compare (default: OUTPUT_DIR/HW2-Exp-*.teIn)")
    p.add_argument("--threshold", type=float, default=0.02,
                   help="Relative MAP threshold (default 0.02)")
    return p.parse_args()


def trec_eval_map(runfile: Path) -> Dict[str, float]:
    cmd = [str(TREC_EVAL), "-q", "-m", "map", str(QREL_PATH), str(runfile)]
    proc = subprocess.run(cmd, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())

    results = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        metric, qid, value = parts
        if metric != "map":
            continue
        results[qid] = float(value)

    return results


def get_output_path(param: Path) -> Path | None:
    try:
        data = json.loads(param.read_text())
    except Exception:
        return None
    for value in data.values():
        if isinstance(value, dict) and value.get("type") == "trec_eval":
            out_path = value.get("outputPath")
            if out_path:
                return Path(out_path)
    return None


def compute_overall_map(per_query_map: Dict[str, float]) -> float:
    values = [v for q, v in per_query_map.items() if q != "all"]
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()

    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        baseline_param = Path(args.baseline_param)
        baseline_path = get_output_path(baseline_param) or Path("HW2-Exp-0.inRank")
    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline file: {baseline_path}")

    if args.runs:
        run_paths = [Path(p) for p in args.runs]
    else:
        run_paths = sorted(Path("OUTPUT_DIR").glob("HW2-Exp-*.teIn"))

    base_map = trec_eval_map(baseline_path)
    if not base_map:
        raise SystemExit("Baseline MAP data is empty.")

    base_overall = compute_overall_map(base_map)

    print(f"\nBaseline MAP: {base_overall:.4f}\n")

    for runfile in run_paths:
        if not runfile.exists():
            continue

        exp_map = trec_eval_map(runfile)
        exp_overall = compute_overall_map(exp_map)

        wins = ties = losses = 0

        print(f"\n=== {runfile.name} ===")
        print(f"Overall MAP: {exp_overall:.4f}")
        print("Per-query MAP comparison:")

        for qid in sorted(base_map.keys()):
            if qid == "all":
                continue

            base = base_map.get(qid, 0.0)
            exp = exp_map.get(qid, 0.0)

            if base > 0:
                rel = (exp - base) / base
            else:
                rel = float('inf') if exp > 0 else 0.0

            if rel >= args.threshold:
                wins += 1
                status = "WIN"
            elif rel <= -args.threshold:
                losses += 1
                status = "LOSS"
            else:
                ties += 1
                status = "TIE"

            print(f"Q{qid}: base={base:.4f} exp={exp:.4f} rel={rel:.3f} → {status}")

        print(f"\nSummary: win {wins} tie {ties} loss {losses}")


if __name__ == "__main__":
    main()
