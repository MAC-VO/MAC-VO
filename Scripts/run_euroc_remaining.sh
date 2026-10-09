#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ODOM_CONFIG="${ODOM_CONFIG:-Config/Experiment/MACVO/MACVO_Fast_local.yaml}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/Results}"
LOG_ROOT="$RESULT_ROOT/logs"
SEQUENCES=(MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203)

mkdir -p "$LOG_ROOT"

for seq in "${SEQUENCES[@]}"; do
    data_config="Config/Sequence/EuRoC_${seq}_local.yaml"
    log_file="$LOG_ROOT/${seq}_batch.log"

    if [[ ! -f "$data_config" ]]; then
        echo "Missing data config: $data_config" >&2
        exit 1
    fi

    {
        echo "===== START ${seq} $(date '+%F %T') ====="
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PYTHON_BIN" MACVO.py \
            --odom "$ODOM_CONFIG" \
            --data "$data_config" \
            --resultRoot "$RESULT_ROOT" \
            --noeval --timing

        latest="$(find "$RESULT_ROOT/MACVO-Fast@${seq}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
            | sort -n | tail -1 | cut -d' ' -f2-)"
        if [[ -z "$latest" || ! -f "$latest/poses.npy" ]]; then
            echo "No completed result found for ${seq}" >&2
            exit 1
        fi

        echo "===== EVALUATE ${seq}: ${latest} ====="
        "$PYTHON_BIN" -m Evaluation.EvalSeq --spaces "$latest"
        echo "===== DONE ${seq} $(date '+%F %T') ====="
    } 2>&1 | tee "$log_file"
done

echo "All sequences completed: ${SEQUENCES[*]}"
