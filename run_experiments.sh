#!/usr/bin/env bash
# run_experiments.sh — Run all HW4 EXP_DIR experiments and evaluate with trec_eval.
#
# Usage:
#   ./run_experiments.sh              # run everything
#   ./run_experiments.sh --skip-rank  # skip inRank generation (reuse existing)
#   ./run_experiments.sh --trec-only  # skip QryEval, only run trec_eval on existing .teIn files

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GENERATOR_PARAM="EXP_DIR/Generate-queries-inRank.param"
INRANK_FILE="EXP_DIR/output/queries.inRank"
EXP_PARAM_DIR="EXP_DIR"
OUTPUT_DIR="OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run_experiments.log"

SKIP_RANK=0
TREC_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-rank) SKIP_RANK=1 ;;
        --trec-only) TREC_ONLY=1; SKIP_RANK=1 ;;
        *) echo "[WARN] Unknown argument: $arg" ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "EXP_DIR/output"
: > "$LOG_FILE"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

run_qryeval() {
    local param="$1"
    local name
    name="$(basename "$param" .param)"
    log "BEGIN QryEval: $name"
    if $PYTHON QryEval.py "$param" 2>&1 | tee -a "$LOG_FILE"; then
        log "END QryEval: $name (ok)"
    else
        log "WARN QryEval: $name (non-zero exit)"
    fi
}

# ── Step 1: Generate shared BM25 inRank file ─────────────────────────────────
if [[ $SKIP_RANK -eq 0 ]]; then
    if [[ -f "$INRANK_FILE" ]]; then
        log "Reusing existing inRank: $INRANK_FILE (use --skip-rank to suppress this message)"
    else
        log "Generating shared inRank: $GENERATOR_PARAM"
        run_qryeval "$GENERATOR_PARAM"
        if [[ ! -f "$INRANK_FILE" ]]; then
            log "ERROR: inRank file not produced at $INRANK_FILE — aborting"
            exit 1
        fi
    fi
fi

# ── Step 2: Run each experiment param file ────────────────────────────────────
if [[ $TREC_ONLY -eq 0 ]]; then
    EXPERIMENTS=(
        EXP_DIR/HW4-Exp-1.1a.param
        EXP_DIR/HW4-Exp-1.1b.param
        EXP_DIR/HW4-Exp-1.1c.param
        EXP_DIR/HW4-Exp-1.1d.param
        EXP_DIR/HW4-Exp-1.2b.param
        EXP_DIR/HW4-Exp-1.2c.param
        EXP_DIR/HW4-Exp-1.2d.param
        EXP_DIR/HW4-Exp-1.3b.param
        EXP_DIR/HW4-Exp-1.3c.param
        EXP_DIR/HW4-Exp-1.3d.param
        EXP_DIR/HW4-Exp-1.4b.param
        EXP_DIR/HW4-Exp-1.4c.param
        EXP_DIR/HW4-Exp-1.4d.param
        EXP_DIR/HW4-Exp-2.1a.param
        EXP_DIR/HW4-Exp-2.1c.param
        EXP_DIR/HW4-Exp-2.1e.param
        EXP_DIR/HW4-Exp-2.2a.param
        EXP_DIR/HW4-Exp-2.2b.param
        EXP_DIR/HW4-Exp-2.2c.param
        EXP_DIR/HW4-Exp-2.2d.param
        EXP_DIR/HW4-Exp-2.2e.param
        EXP_DIR/HW4-Exp-2.2f.param
        EXP_DIR/HW4-Exp-2.3a.param
        EXP_DIR/HW4-Exp-2.3b.param
        EXP_DIR/HW4-Exp-2.3c.param
        EXP_DIR/HW4-Exp-2.3d.param
        EXP_DIR/HW4-Exp-2.3e.param
        EXP_DIR/HW4-Exp-2.3f.param
    )

    TOTAL=${#EXPERIMENTS[@]}
    COUNT=0
    for param in "${EXPERIMENTS[@]}"; do
        COUNT=$((COUNT + 1))
        if [[ ! -f "$param" ]]; then
            log "SKIP ($COUNT/$TOTAL): $param not found"
            continue
        fi
        log "=== ($COUNT/$TOTAL) ==="
        run_qryeval "$param"
    done
fi

# ── Step 3: Run trec_eval via run_tests.py on all produced .teIn files ────────
log "=== Running trec_eval on all HW4-Exp-*.teIn files ==="

TEIN_FILES=()
for f in "$OUTPUT_DIR"/HW4-Exp-*.teIn; do
    [[ -f "$f" ]] && TEIN_FILES+=("$f")
done

if [[ ${#TEIN_FILES[@]} -eq 0 ]]; then
    log "ERROR: No HW4-Exp-*.teIn files found in $OUTPUT_DIR"
    exit 1
fi

log "Found ${#TEIN_FILES[@]} .teIn files — running trec_eval"
$PYTHON run_tests.py "${TEIN_FILES[@]}" 2>&1 | tee -a "$LOG_FILE"

log "=== All done. Logs: $LOG_FILE ==="
