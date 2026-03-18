#!/usr/bin/env python3
"""
Run EXP_DIR experiments and optional trec_eval summaries.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO, Tuple


QREL_PATH = Path("INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed")
TREC_EVAL = Path("INPUT_DIR/trec_eval-9.0.4")
DEFAULT_PARAM_DIR = Path("EXP_DIR/params")
DEFAULT_OUTPUT_DIR = Path("EXP_DIR/output")
DEFAULT_GENERATOR_PARAM = DEFAULT_PARAM_DIR / "Generate-queries-inRank.param"
DEFAULT_LOG_FILE = DEFAULT_OUTPUT_DIR / "run_exp_dir_experiments.txt"

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
    p.add_argument("--run", action="store_true", help="Run QryEval for the selected experiments")
    p.add_argument("--trec", action="store_true", help="Run trec_eval for the selected experiments")
    p.add_argument(
        "--conda-env",
        default=None,
        help="Optional conda env for QryEval. By default, use the current Python interpreter.",
    )
    p.add_argument("--param-dir", default=str(DEFAULT_PARAM_DIR), help="Directory with Exp-*.param files")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write .teOut files")
    p.add_argument(
        "--generator-param",
        default=str(DEFAULT_GENERATOR_PARAM),
        help="Param file that generates the shared queries.inRank baseline",
    )
    p.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Write detailed execution logs to this .txt file.",
    )
    p.add_argument(
        "--force-generator",
        action="store_true",
        help="Regenerate queries.inRank even if it already exists.",
    )
    p.add_argument(
        "cases",
        nargs="*",
        help="Experiment IDs such as 1.1b or Exp-1.1b. Omit to run all experiments.",
    )
    return p.parse_args()


def normalize_case(case: str) -> str:
    case = case.strip()
    return case if case.startswith("Exp-") else f"Exp-{case}"


def case_sort_key(case: str) -> Tuple[int, int, str]:
    m = re.fullmatch(r"Exp-(\d+)\.(\d+)([a-z])", case)
    if m:
        return (int(m.group(1)), int(m.group(2)), m.group(3))
    return (999, 999, case)


def discover_cases(param_dir: Path) -> List[str]:
    cases = []
    for f in sorted(param_dir.glob("Exp-*.param")):
        cases.append(f.stem)
    return sorted(cases, key=case_sort_key)


def case_to_param(case: str, param_dir: Path) -> Path:
    return param_dir / f"{normalize_case(case)}.param"


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
    return DEFAULT_OUTPUT_DIR / f"{normalize_case(case)}.teIn"


def get_generator_runfile(generator_param: Path) -> Path:
    out_path = get_output_path(generator_param)
    if out_path is not None:
        return out_path
    return DEFAULT_OUTPUT_DIR / "queries.inRank"


def log_line(handle: TextIO, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"[{timestamp}] {message}\n")
    handle.flush()


def run_command(cmd: List[str], log_handle: TextIO, label: str) -> int:
    log_line(log_handle, f"BEGIN {label}")
    log_line(log_handle, f"COMMAND {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        log_handle.write(line)
    proc.stdout.close()
    rc = proc.wait()

    log_line(log_handle, f"END {label} (exit {rc})")
    log_handle.write("\n")
    log_handle.flush()
    return rc


def run_qryeval(param: Path, conda_env: Optional[str], log_handle: TextIO) -> int:
    current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env and current_conda_env != conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python", "QryEval.py", str(param)]
    else:
        cmd = [sys.executable, "QryEval.py", str(param)]
    return run_command(cmd, log_handle, f"QryEval {param}")


def run_trec_eval(runfile: Path, outpath: Path, log_handle: TextIO) -> int:
    cmd = [str(TREC_EVAL), *METRICS, str(QREL_PATH), str(runfile)]
    log_line(log_handle, f"BEGIN trec_eval {runfile}")
    log_line(log_handle, f"COMMAND {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout:
        outpath.write_text(proc.stdout)
        log_handle.write(proc.stdout)
    if proc.stderr:
        log_handle.write(proc.stderr)
    log_line(log_handle, f"END trec_eval {runfile} (exit {proc.returncode})")
    log_handle.write("\n")
    log_handle.flush()
    return proc.returncode


def main() -> None:
    args = parse_args()
    param_dir = Path(args.param_dir)
    output_dir = Path(args.output_dir)
    generator_param = Path(args.generator_param)
    log_file = Path(args.log_file)

    if not args.run and not args.trec:
        args.run = True
        args.trec = True

    cases = [normalize_case(c) for c in (args.cases or discover_cases(param_dir))]
    if not cases:
        raise SystemExit("No Exp-*.param files found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing detailed logs to {log_file}")

    with log_file.open("w") as log_handle:
        log_line(log_handle, "START run_exp_dir_experiments.py")
        log_line(log_handle, f"Working directory: {Path.cwd()}")
        log_line(log_handle, f"Cases: {', '.join(cases)}")
        log_line(log_handle, f"Run QryEval: {args.run}")
        log_line(log_handle, f"Run trec_eval: {args.trec}")
        log_line(log_handle, f"Conda env override: {args.conda_env}")
        log_line(log_handle, f"Generator param: {generator_param}")
        log_line(log_handle, f"Log file: {log_file}")
        log_handle.write("\n")

        if args.run:
            if not generator_param.exists():
                log_line(log_handle, f"ERROR Missing generator param: {generator_param}")
                raise SystemExit(f"Missing generator param: {generator_param}")

            generator_runfile = get_generator_runfile(generator_param)
            if generator_runfile.exists() and not args.force_generator:
                log_line(log_handle, f"Using existing shared inRank: {generator_runfile}")
            else:
                log_line(log_handle, f"Generating shared inRank with {generator_param}")
                rc = run_qryeval(generator_param, args.conda_env, log_handle)
                if rc != 0:
                    log_line(log_handle, f"ERROR Failed to generate queries.inRank (exit {rc})")
                    raise SystemExit(f"Failed to generate queries.inRank (exit {rc})")

            for case in cases:
                param = case_to_param(case, param_dir)
                if not param.exists():
                    log_line(log_handle, f"Skip {case} (missing {param})")
                    continue
                log_line(log_handle, f"Running {param}")
                rc = run_qryeval(param, args.conda_env, log_handle)
                if rc != 0:
                    log_line(log_handle, f"[WARN] QryEval failed for {case} (exit {rc})")

        if args.trec:
            for case in cases:
                runfile = case_to_runfile(case, param_dir)
                if not runfile.exists():
                    log_line(log_handle, f"Skip {case} (missing {runfile})")
                    continue
                outpath = output_dir / f"{normalize_case(case)}.teOut"
                log_line(log_handle, f"trec_eval {case}")
                rc = run_trec_eval(runfile, outpath, log_handle)
                if rc != 0:
                    log_line(log_handle, f"[WARN] trec_eval failed for {case} (exit {rc})")

        log_line(log_handle, "END run_exp_dir_experiments.py")

    print(f"Finished. Detailed logs are in {log_file}")


if __name__ == "__main__":
    main()
