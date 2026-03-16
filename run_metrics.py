"""
Run trec_eval on a .teIn file and write selected metrics to a .teOut file.

Usage examples
--------------
  python run_metrics.py --run OUTPUT_DIR/HW3-Exp-1.teIn
  python run_metrics.py --qrels TEST_DIR/HW3-train.qrels --run OUTPUT_DIR/HW3-Exp-1.teIn --out OUTPUT_DIR/HW3-Exp-1.teOut
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qrels",
        default="TEST_DIR/HW3-train.qrels",
        help="Path to qrels file (default: TEST_DIR/HW3-train.qrels)",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to .teIn run file (trec_eval input)",
    )
    parser.add_argument(
        "--out",
        help="Path to write .teOut metrics file "
             "(default: same as --run, with .teOut extension)",
    )
    args = parser.parse_args()

    qrels_path = args.qrels
    run_path = args.run
    out_path = args.out

    if out_path is None:
        root, _ = os.path.splitext(run_path)
        out_path = root + ".teOut"

    # trec_eval metrics to compute:
    # - MRR (recip_rank)
    # - P@10, P@20, P@30
    # - MAP@1K (map_cut.1000)
    # - NDCG@10, NDCG@20, NDCG@30
    metrics = [
        "-m", "recip_rank",
        "-m", "P.10",
        "-m", "P.20",
        "-m", "P.30",
        "-m", "map_cut.1000",
        "-m", "ndcg_cut.10",
        "-m", "ndcg_cut.20",
        "-m", "ndcg_cut.30",
    ]

    cmd = ["trec_eval"] + metrics + [qrels_path, run_path]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("Error: trec_eval not found on PATH. Install trec_eval and try again.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Error: trec_eval failed:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    # Write metrics to the .teOut file.
    with open(out_path, "w") as f:
        f.write(result.stdout)

    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()

