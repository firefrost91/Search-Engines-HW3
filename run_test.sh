#!/bin/bash
#
# Effective local testing for QryEval (Option 1 - fastest).
# Run from the QryEval directory:  ./run_test.sh [options]
#
# Usage:
#   ./run_test.sh              # Run trec_eval on all existing OUTPUT_DIR/*.teIn, save to OUTPUT_DIR/*.teOut
#   ./run_test.sh --run        # First run QryEval for each TEST_DIR/*.param, then run trec_eval
#   ./run_test.sh --diff       # After trec_eval, diff your OUTPUT_DIR/*.teOut vs TEST_DIR reference
#   ./run_test.sh --run --diff # Full: run QryEval, trec_eval, then diff
#   ./run_test.sh 0 5 16       # Run trec_eval only for test cases HW1-Train-0, 5, 16
#

set -e
QRELVAL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC_EVAL="./INPUT_DIR/trec_eval-9.0.4"
METRICS="-m num_q -m num_ret -m num_rel -m num_rel_ret -m map -m recip_rank -m P -m ndcg -m ndcg_cut -m recall.100,500,1000"

RUN_QRYEVAL=false
RUN_DIFF=false
CASES=()

for arg in "$@"; do
  case "$arg" in
    --run)  RUN_QRYEVAL=true ;;
    --diff) RUN_DIFF=true ;;
    *)      CASES+=("$arg") ;;
  esac
done

# If specific case numbers given (e.g. 0 5 16), use those; else all param files.
if [ ${#CASES[@]} -eq 0 ]; then
  for f in TEST_DIR/HW1-Train-*.param; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .param)
    num=${base#HW1-Train-}
    CASES+=("$num")
  done
fi

mkdir -p OUTPUT_DIR

# Step 1: Optionally run QryEval for each test case to produce .teIn
if [ "$RUN_QRYEVAL" = true ]; then
  echo "=== Running QryEval for each test case ==="
  for num in "${CASES[@]}"; do
    param="TEST_DIR/HW1-Train-${num}.param"
    if [ -f "$param" ]; then
      echo "  QryEval $param"
      python QryEval.py "$param" || true
    fi
  done
fi

# Step 2: Run trec_eval on each .teIn, store output in .teOut
echo "=== Running trec_eval (your results -> OUTPUT_DIR/*.teOut) ==="
for num in "${CASES[@]}"; do
  teIn="OUTPUT_DIR/HW1-Train-${num}.teIn"
  teOut="OUTPUT_DIR/HW1-Train-${num}.teOut"
  if [ -f "$teIn" ]; then
    echo "  trec_eval HW1-Train-${num}"
    $TREC_EVAL $METRICS "$QRELVAL" "$teIn" > "$teOut"
  else
    echo "  Skip HW1-Train-${num} (no $teIn; run with --run to generate)"
  fi
done

# Step 3: Optionally diff against reference .teOut in TEST_DIR
# Compare only lines with "all" (aggregate metrics); some trec_eval versions omit per-query output.
if [ "$RUN_DIFF" = true ]; then
  echo "=== Diff: your OUTPUT_DIR/*.teOut vs reference TEST_DIR/*.teOut (aggregate 'all' metrics) ==="
  for num in "${CASES[@]}"; do
    ref="TEST_DIR/HW1-Train-${num}.teOut"
    out="OUTPUT_DIR/HW1-Train-${num}.teOut"
    if [ -f "$out" ] && [ -f "$ref" ]; then
      if diff -q <(grep $'\tall\t' "$out" | sort) <(grep $'\tall\t' "$ref" | sort) > /dev/null 2>&1; then
        echo "  HW1-Train-${num}: MATCH (aggregate metrics)"
      else
        echo "  HW1-Train-${num}: DIFFER"
        diff <(grep $'\tall\t' "$ref" | sort) <(grep $'\tall\t' "$out" | sort) || true
      fi
    fi
  done
fi

echo "Done."
