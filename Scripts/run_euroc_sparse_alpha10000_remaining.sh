#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="${WORKTREE:-$SCRIPT_ROOT}"
PYTHON="${PYTHON:-/home/zzh/MACVO/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/home/zzh/MACVO/Results/SparseORBLoop_EuRoC_alpha10000}"
MODE="${MODE:-window_orb_loop_sparse}"
ODOM_CONFIG="${ODOM_CONFIG:-$WORKTREE/Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse.yaml}"
DEFAULT_SEQUENCES=(MH01 MH02 MH03 MH05 V101 V102 V103 V201 V202)
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "--list-defaults" ]]; then
    echo "${DEFAULT_SEQUENCES[*]}"
    exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0") [--dry-run] [SEQUENCE ...]"
    echo "Defaults: ${DEFAULT_SEQUENCES[*]}"
    exit 0
fi

if (( $# )); then
    SEQUENCES=("$@")
else
    SEQUENCES=("${DEFAULT_SEQUENCES[@]}")
fi

STATE_DIR="$RESULT_ROOT/.batch_state"
LOG_DIR="$RESULT_ROOT/logs"
BATCH_LOG="$RESULT_ROOT/batch.log"
SUMMARY="$RESULT_ROOT/summary.tsv"

emit() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo -e "$*"
    else
        echo -e "$*" | tee -a "$BATCH_LOG"
    fi
}

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$STATE_DIR" "$LOG_DIR"
fi

missing=0
required=(
    "$PYTHON"
    "$WORKTREE/Scripts/Experiment/CompareOnlineORBLoop.py"
    "$ODOM_CONFIG"
    "$WORKTREE/build/orb_bow/macvo_orb_bow"
    "$WORKTREE/cache/ORBvoc.txt"
    "$WORKTREE/Model/MACVO_FrontendCov.pth"
)
for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        emit "MISSING\t$path"
        missing=1
    fi
done
if [[ ! -x "$PYTHON" ]]; then
    emit "MISSING\texecutable Python: $PYTHON"
    missing=1
fi
if [[ ! -x "$WORKTREE/build/orb_bow/macvo_orb_bow" ]]; then
    emit "MISSING\texecutable sidecar: $WORKTREE/build/orb_bow/macvo_orb_bow"
    missing=1
fi

for sequence in "${SEQUENCES[@]}"; do
    config="$WORKTREE/Config/Sequence/EuRoC_${sequence}_local.yaml"
    if [[ ! -f "$config" ]]; then
        emit "MISSING\t$config"
        missing=1
        continue
    fi
    data_root="$($PYTHON - "$config" <<'PY'
from pathlib import Path
import sys, yaml
with Path(sys.argv[1]).open() as stream:
    config = yaml.safe_load(stream)
print(config["args"]["root"])
PY
)"
    if [[ ! -d "$data_root" ]]; then
        emit "MISSING\t$data_root"
        missing=1
    fi
done
if (( missing )); then
    exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
    exec 9>"$RESULT_ROOT/.batch.lock"
    if ! flock -n 9; then
        echo -e "LOCKED\t$RESULT_ROOT"
        exit 3
    fi
fi

completed_result() {
    "$PYTHON" - "$RESULT_ROOT" "$1" "$MODE" <<'PY'
from pathlib import Path
import json, sys
root, sequence, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
matches = []
for path in root.rglob("metrics.json") if root.exists() else ():
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(rows, list):
        continue
    if any(
        isinstance(row, dict)
        and row.get("sequence") == sequence
        and row.get("mode") == mode
        for row in rows
    ):
        matches.append(path)
if matches:
    newest = max(matches, key=lambda path: path.stat().st_mtime)
    print(newest.parent)
PY
}

summary_row() {
    "$PYTHON" - "$1" "$2" "$MODE" <<'PY'
from pathlib import Path
import json, sys
folder, sequence, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
if not isinstance(rows, list):
    raise ValueError("metrics payload must be a list")
row = next(
    item for item in rows
    if isinstance(item, dict)
    and item.get("sequence") == sequence
    and item.get("mode") == mode
)
values = [
    row.get("RMSE_ATE", ""), row.get("RMSE_RTE", ""),
    row.get("RMSE_ROE", ""), row.get("RMSE_RPE", ""),
    row.get("runtime_mean_ms", ""), row.get("accepted_loops", ""),
    row.get("effective_long_loops_001", ""), row.get("peak_vram_bytes", ""),
]
print("\t".join(str(value) for value in values))
PY
}

if [[ "$DRY_RUN" != "1" && ! -s "$SUMMARY" ]]; then
    printf "sequence\tstatus\tresult_dir\tRMSE_ATE\tRMSE_RTE\tRMSE_ROE\tRMSE_RPE\truntime_mean_ms\tloop_factors\teffective_long_loops_001\tpeak_vram_bytes\n" > "$SUMMARY"
fi

successful=()
skipped=()
failed=()
cd "$WORKTREE" || exit 2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

emit "START\t$(date --iso-8601=seconds)\t${SEQUENCES[*]}"
for sequence in "${SEQUENCES[@]}"; do
    complete_marker="$STATE_DIR/${sequence}.complete"
    failed_marker="$STATE_DIR/${sequence}.failed"
    existing="$(completed_result "$sequence")"
    if [[ -f "$complete_marker" && -n "$existing" ]]; then
        emit "SKIP\t$sequence\t$existing"
        if [[ "$DRY_RUN" != "1" ]]; then
            printf "%s\tskipped\t%s\t\t\t\t\t\t\t\t\n" "$sequence" "$existing" >> "$SUMMARY"
        fi
        skipped+=("$sequence")
        continue
    fi

    emit "RUN\t$sequence"
    log_file="$LOG_DIR/${sequence}.log"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "$PYTHON Scripts/Experiment/CompareOnlineORBLoop.py --sequence $sequence --seq-from 0 --seed 0 --modes $MODE --result-root $RESULT_ROOT"
        if [[ "${DRY_RUN_FAIL_SEQUENCE:-}" == "$sequence" ]]; then
            emit "FAIL\t$sequence\tdry-run"
            failed+=("$sequence")
            continue
        fi
        successful+=("$sequence")
        continue
    fi

    rm -f "$complete_marker" "$failed_marker"
    if "$PYTHON" Scripts/Experiment/CompareOnlineORBLoop.py \
        --sequence "$sequence" --seq-from 0 --seed 0 \
        --modes "$MODE" --result-root "$RESULT_ROOT" \
        2>&1 | tee -a "$log_file" "$BATCH_LOG"; then
        result="$(completed_result "$sequence")"
        if [[ -z "$result" ]]; then
            emit "FAIL\t$sequence\tmissing metrics"
            marker_tmp="$(mktemp "$STATE_DIR/.${sequence}.failed.XXXXXX")"
            echo "missing metrics $(date --iso-8601=seconds)" > "$marker_tmp"
            mv "$marker_tmp" "$failed_marker"
            printf "%s\tfailed\t\t\t\t\t\t\t\t\t\n" "$sequence" >> "$SUMMARY"
            failed+=("$sequence")
            continue
        fi
        metrics="$(summary_row "$result" "$sequence")"
        printf "%s\tcomplete\t%s\t%s\n" "$sequence" "$result" "$metrics" >> "$SUMMARY"
        marker_tmp="$(mktemp "$STATE_DIR/.${sequence}.complete.XXXXXX")"
        echo "$result" > "$marker_tmp"
        mv "$marker_tmp" "$complete_marker"
        emit "DONE\t$sequence\t$result"
        successful+=("$sequence")
    else
        marker_tmp="$(mktemp "$STATE_DIR/.${sequence}.failed.XXXXXX")"
        date --iso-8601=seconds > "$marker_tmp"
        mv "$marker_tmp" "$failed_marker"
        printf "%s\tfailed\t\t\t\t\t\t\t\t\t\n" "$sequence" >> "$SUMMARY"
        emit "FAIL\t$sequence"
        failed+=("$sequence")
    fi
done

emit "SUCCESS\t${successful[*]:-none}"
emit "SKIPPED\t${skipped[*]:-none}"
emit "FAILED\t${failed[*]:-none}"
emit "END\t$(date --iso-8601=seconds)"

if (( ${#failed[@]} )); then
    exit 1
fi
