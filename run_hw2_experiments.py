#!/usr/bin/env python3
"""
Run HW2 experiments and optional trec_eval summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional


QREL_PATH = Path("INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed")
TREC_EVAL = Path("INPUT_DIR/trec_eval-9.0.4")
DEFAULT_PARAM_DIR = Path("TEST_DIR")
DEFAULT_OUTPUT_DIR = Path("OUTPUT_DIR")

METRICS = [
    "-m", "num_q",
    "-m", "num_ret",
    "-m", "num_rel",
    "-m", "num_rel_ret",
    "-m", "map",
    "-m", "recip_rank",
    "-m", "P",
    "-m", "ndcg",
    "-m", "ndcg_cut",
    "-m", "recall.100,500,1000",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--run", action="store_true", help="Run QryEval for each selected experiment")
    p.add_argument("--trec", action="store_true", help="Run trec_eval for each selected experiment")
    p.add_argument("--conda-env", default="11x42-26S-a", help="Conda env for QryEval")
    p.add_argument("--param-dir", default=str(DEFAULT_PARAM_DIR), help="Directory with HW2-Exp-*.param files")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write trec_eval outputs")
    p.add_argument("cases", nargs="*", help="Experiment IDs (e.g., 1.1a 2.2f). Omit for all.")
    return p.parse_args()


def discover_cases(param_dir: Path) -> List[str]:
    cases = []
    for f in sorted(param_dir.glob("HW2-Exp-*.param")):
        name = f.stem.replace("HW2-Exp-", "")
        cases.append(name)
    return cases


def case_to_param(case: str, param_dir: Path) -> Path:
    return param_dir / f"HW2-Exp-{case}.param"


def get_output_path(param: Path) -> Optional[Path]:
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


def case_to_runfile(case: str, param_dir: Path) -> Path:
    param = case_to_param(case, param_dir)
    out_path = get_output_path(param)
    if out_path is not None:
        return out_path
    if case == "0":
        return Path("HW2-Exp-0.inRank")
    return DEFAULT_OUTPUT_DIR / f"HW2-Exp-{case}.teIn"


def run_qryeval(param: Path, conda_env: str) -> int:
    cmd = ["conda", "run", "-n", conda_env, "python", "QryEval.py", str(param)]
    return subprocess.run(cmd, check=False).returncode


def run_trec_eval(runfile: Path, outpath: Path) -> int:
    cmd = [str(TREC_EVAL), *METRICS, str(QREL_PATH), str(runfile)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(proc.stderr, end="")
        return proc.returncode
    outpath.write_text(proc.stdout)
    return 0


def main() -> None:
    args = parse_args()
    param_dir = Path(args.param_dir)
    output_dir = Path(args.output_dir)
    cases = args.cases or discover_cases(param_dir)
    if not cases:
        raise SystemExit("No HW2-Exp-*.param files found.")

    # Ensure baseline exists first when running PRF experiments.
    baseline_param = case_to_param("0", param_dir)
    baseline_run = case_to_runfile("0", param_dir)
    if args.run and ("0" not in cases) and not baseline_run.exists():
        if baseline_param.exists():
            print("Baseline run missing; generating HW2-Exp-0.inRank first.")
            run_qryeval(baseline_param, args.conda_env)

    if args.run:
        for case in cases:
            param = case_to_param(case, param_dir)
            if not param.exists():
                print(f"Skip {case} (missing {param})")
                continue
            print(f"Running {param}")
            rc = run_qryeval(param, args.conda_env)
            if rc != 0:
                print(f"  [WARN] QryEval failed for {case} (exit {rc})")

    if args.trec:
        output_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            runfile = case_to_runfile(case, param_dir)
            if not runfile.exists():
                print(f"Skip {case} (missing {runfile})")
                continue
            outpath = output_dir / f"HW2-Exp-{case}.teOut"
            print(f"trec_eval {case}")
            rc = run_trec_eval(runfile, outpath)
            if rc != 0:
                print(f"  [WARN] trec_eval failed for {case} (exit {rc})")


if __name__ == "__main__":
    main()
