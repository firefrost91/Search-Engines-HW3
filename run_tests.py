#!/usr/bin/env python3
"""
Effective local testing for QryEval (Python version).

Run from the QryEval directory:
  ./run_tests.py                    # trec_eval on existing OUTPUT_DIR/*.teIn -> OUTPUT_DIR/*.teout
  ./run_tests.py --run              # run QryEval for each TEST_DIR/*.param, then trec_eval
  ./run_tests.py --diff             # after trec_eval, diff OUTPUT_DIR/*.teout vs TEST_DIR reference (aggregate 'all' only)
  ./run_tests.py --run --diff       # full: QryEval -> trec_eval -> diff
  ./run_tests.py 0 5 16             # only cases 0, 5, 16 (HW1-Train-0, etc.)
  ./run_tests.py OUTPUT_DIR/HW2-Exp-1.1a.teIn

  # Compare your output to professor's reference in hw3-tests:
  ./run_tests.py --ref-dir hw3-tests --diff   # diff OUTPUT_DIR vs hw3-tests (HW3-Train-*.teOut)
  ./run_tests.py --ref-dir hw3-tests --diff 0 1 2   # only cases 0, 1, 2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


QRELVAL = Path("INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed")
TREC_EVAL = Path("INPUT_DIR/trec_eval-9.0.4")

METRICS = [
    "-m", "num_ret",
    "-m", "num_rel",
    "-m", "num_rel_ret",
    "-m", "map",
    "-m", "recip_rank",
    "-m", "P.10,20,30",
    "-m", "ndcg",
    "-m", "ndcg_cut.10,20,30",
]

# Display order for printing (label -> trec_eval key).
METRIC_ORDER = [
    ("num_ret",   "num_ret"),
    ("num_rel",   "num_rel"),
    ("num_rel_ret", "num_rel_ret"),
    ("MAP",       "map"),
    ("MRR",       "recip_rank"),
    ("P@10",      "P_10"),
    ("P@20",      "P_20"),
    ("P@30",      "P_30"),
    ("NDCG",      "ndcg"),
    ("NDCG@10",   "ndcg_cut_10"),
    ("NDCG@20",   "ndcg_cut_20"),
    ("NDCG@30",   "ndcg_cut_30"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--run", action="store_true", help="Run QryEval for each selected case before trec_eval")
    p.add_argument("--diff", action="store_true", help="Diff aggregate 'all' metrics vs reference")
    p.add_argument("--ref-dir", default=None, help="Reference directory (e.g. hw3-tests). If set, discover HW3-Train-*.teOut and diff vs OUTPUT_DIR")
    p.add_argument("--log", default="OUTPUT_DIR/run_test_log.txt", help="Path to log file")
    p.add_argument("cases", nargs="*", help="Case numbers (e.g., 0 5 16). If omitted, run all discovered cases")
    return p.parse_args()


def discover_cases(ref_dir: Path | None = None) -> Tuple[List[str], str]:
    """Return (case_numbers, prefix). prefix is 'HW1-Train' or 'HW3-Train'."""
    if ref_dir is not None:
        ref_dir = Path(ref_dir)
        cases: List[str] = []
        for f in sorted(ref_dir.glob("HW3-Train-*.teOut")):
            m = re.match(r"HW3-Train-(\d+)\.teOut$", f.name)
            if m:
                cases.append(m.group(1))
        return (cases, "HW3-Train")
    cases = []
    for f in sorted(Path("TEST_DIR").glob("HW1-Train-*.param")):
        m = re.match(r"HW1-Train-(\d+)\.param$", f.name)
        if m:
            cases.append(m.group(1))
    return (cases, "HW1-Train")


def run_subprocess_and_stream(cmd: List[str]) -> int:
    """
    Run a subprocess, capture stdout/stderr, and print them so they go to console + Tee log.
    Returns process return code.
    """
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def run_qryeval(cases: List[str]) -> None:
    print("=== Running QryEval for each test case ===")
    for num in cases:
        param = Path(f"TEST_DIR/HW1-Train-{num}.param")
        if not param.exists():
            continue

        print(f"  QryEval {param}")
        rc = run_subprocess_and_stream([sys.executable, "QryEval.py", str(param)])
        if rc != 0:
            print(f"    [WARN] QryEval failed for case {num} (exit {rc})")


def run_trec_eval(cases: List[str], direct_inputs: List[Path]) -> None:
    print("=== Running trec_eval (your results -> OUTPUT_DIR/*.teout) ===")
    Path("OUTPUT_DIR").mkdir(parents=True, exist_ok=True)

    for num in cases:
        te_in = Path(f"OUTPUT_DIR/HW1-Train-{num}.teIn")
        te_out = Path(f"OUTPUT_DIR/HW1-Train-{num}.teout")

        if not te_in.exists():
            print(f"  Skip HW1-Train-{num} (no {te_in}; run with --run to generate)")
            continue

        print(f"  trec_eval HW1-Train-{num}")
        cmd = [str(TREC_EVAL), *METRICS, str(QRELVAL), str(te_in)]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.returncode != 0:
            print(f"    [ERROR] trec_eval failed for case {num}")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            continue

        write_selected_metrics(te_out, proc.stdout)

    for te_in in direct_inputs:
        if not te_in.exists():
            print(f"  Skip {te_in} (file not found)")
            continue

        if te_in.suffix.lower() == ".tein":
            te_out = te_in.with_suffix(".teout")
        else:
            te_out = Path(f"{te_in}.teout")

        print(f"  trec_eval {te_in}")
        cmd = [str(TREC_EVAL), *METRICS, str(QRELVAL), str(te_in)]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.returncode != 0:
            print(f"    [ERROR] trec_eval failed for {te_in}")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            continue

        write_selected_metrics(te_out, proc.stdout)


def write_selected_metrics(out_path: Path, trec_output: str) -> None:
    """Write trec_eval output lines directly (same format as professor's .teOut)."""
    # Keep all lines; filter to only the metrics we care about for display.
    all_lines = [l for l in trec_output.splitlines() if l.strip()]
    out_path.write_text("\n".join(all_lines) + "\n")
    print(f"    wrote {out_path}")

    # Print a summary of 'all' metrics for the configured METRIC_ORDER.
    values = {}
    for line in all_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "all":
            values[parts[0]] = parts[2]
    for label, trec_key in METRIC_ORDER:
        val = values.get(trec_key, "NA")
        print(f"  {label}\t{val}")


def resolve_inputs(raw_cases: List[str]) -> Tuple[List[str], List[Path]]:
    case_nums: List[str] = []
    direct_inputs: List[Path] = []

    for token in raw_cases:
        if token.isdigit():
            case_nums.append(str(int(token)))
            continue

        p = Path(token)
        if p.exists():
            direct_inputs.append(p)
            continue

        print(f"[WARN] Ignoring unknown case/input: {token}")

    return case_nums, direct_inputs


def extract_all_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted([line for line in path.read_text().splitlines() if "\tall\t" in line])


def _parse_teout(path: Path) -> dict:
    """Parse a .teOut/.teout file (metric  qid  value) into trec_key -> value for 'all'."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "all":
            out[parts[0]] = parts[2]
    return out


def parse_ref_teout(path: Path) -> dict:
    return _parse_teout(path)


def parse_our_teout(path: Path) -> dict:
    return _parse_teout(path)


def diff_hw3_metrics(ref_dir: Path, cases: List[str], out_dir: Path = Path("OUTPUT_DIR")) -> None:
    """Diff OUTPUT_DIR/HW3-Train-{n}.teout vs ref_dir/HW3-Train-{n}.teOut (metric values)."""
    print(f"=== Diff: your {out_dir}/HW3-Train-*.teout vs reference {ref_dir}/HW3-Train-*.teOut ===")
    ref_dir = Path(ref_dir)
    for num in cases:
        ref_file = ref_dir / f"HW3-Train-{num}.teOut"
        out_file = out_dir / f"HW3-Train-{num}.teout"
        if not ref_file.exists():
            print(f"  HW3-Train-{num}: skip (no reference {ref_file})")
            continue
        if not out_file.exists():
            print(f"  HW3-Train-{num}: MISSING your output {out_file}")
            continue
        ref_vals = parse_ref_teout(ref_file)
        our_vals = parse_our_teout(out_file)
        all_keys = sorted(set(ref_vals) | set(our_vals))
        diffs = []
        for k in all_keys:
            r, o = ref_vals.get(k), our_vals.get(k)
            if r != o:
                diffs.append((k, r, o))
        if not diffs:
            print(f"  HW3-Train-{num}: MATCH")
        else:
            print(f"  HW3-Train-{num}: DIFFER")
            for k, r, o in diffs[:20]:
                print(f"    {k}: ref={r} yours={o}")
            if len(diffs) > 20:
                print(f"    ... ({len(diffs) - 20} more)")


def diff_all_metrics(cases: List[str]) -> None:
    print("=== Diff: your OUTPUT_DIR/*.teout vs reference TEST_DIR/*.teOut (aggregate 'all' metrics) ===")
    for num in cases:
        ref = Path(f"TEST_DIR/HW1-Train-{num}.teOut")
        out = Path(f"OUTPUT_DIR/HW1-Train-{num}.teout")

        if not ref.exists() or not out.exists():
            continue

        ref_all = extract_all_lines(ref)
        out_all = extract_all_lines(out)

        ref_set = set(ref_all)
        out_set = set(out_all)

        missing = sorted(ref_set - out_set)
        extra = sorted(out_set - ref_set)

        # MATCH if every reference metric line appears in your output (same name + value).
        # Reference files may list only a subset of metrics; your trec_eval may output more.
        if not missing:
            print(f"  HW1-Train-{num}: MATCH (aggregate metrics)")
            continue

        print(f"  HW1-Train-{num}: DIFFER")

        if missing:
            print("    Missing (in reference but not in your output):")
            for line in missing[:50]:
                print(f"      - {line}")
            if len(missing) > 50:
                print(f"      ... ({len(missing) - 50} more)")

        if extra:
            print("    Extra (in your output but not in reference):")
            for line in extra[:50]:
                print(f"      + {line}")
            if len(extra) > 50:
                print(f"      ... ({len(extra) - 50} more)")


def main() -> None:
    args = parse_args()
    ref_dir = Path(args.ref_dir) if args.ref_dir else None

    # Ensure OUTPUT_DIR exists before opening the log file
    Path("OUTPUT_DIR").mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        with log_path.open("w") as log_file:
            sys.stdout = Tee(orig_out, log_file)
            sys.stderr = Tee(orig_err, log_file)

            # Resolve cases and/or direct input files
            if ref_dir is not None:
                all_cases, prefix = discover_cases(ref_dir)
                if args.cases:
                    requested = [t for t in args.cases if t.isdigit()]
                    cases = [c for c in all_cases if c in requested] if requested else all_cases
                else:
                    cases = all_cases
                direct_inputs = []
            elif args.cases:
                cases, direct_inputs = resolve_inputs(args.cases)
            else:
                cases, _ = discover_cases(None)
                direct_inputs = []

            if not cases and not direct_inputs:
                print("[ERROR] No valid cases or input files found.")
                sys.exit(2)

            # HW3: compare your OUTPUT_DIR vs professor's ref_dir (diff only)
            if ref_dir is not None:
                if args.diff:
                    diff_hw3_metrics(ref_dir, cases)
                else:
                    print(f"Found {len(cases)} HW3-Train cases in {ref_dir}. Use --diff to compare your OUTPUT_DIR/*.teout to reference.")
                print("Done.")
                return

            # Basic sanity checks (HW1 path)
            if not QRELVAL.exists():
                print(f"[ERROR] Missing qrels file: {QRELVAL}")
                sys.exit(2)
            if not TREC_EVAL.exists():
                print(f"[ERROR] Missing trec_eval executable: {TREC_EVAL}")
                sys.exit(2)

            # Step 1
            if args.run:
                if cases:
                    run_qryeval(cases)
                else:
                    print("[WARN] --run ignored because no HW1-Train numeric cases were provided.")

            # Step 2
            run_trec_eval(cases, direct_inputs)

            # Step 3
            if args.diff:
                if cases:
                    diff_all_metrics(cases)
                else:
                    print("[WARN] --diff ignored because no HW1-Train numeric cases were provided.")

            print("Done.")
    finally:
        # restore
        sys.stdout, sys.stderr = orig_out, orig_err


if __name__ == "__main__":
    main()
    # Test case from the PDF
