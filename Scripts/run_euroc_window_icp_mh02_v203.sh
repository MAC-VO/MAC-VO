#!/usr/bin/env bash
set -Eeuo pipefail

MACVO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MACVO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$MACVO_ROOT/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$MACVO_ROOT/Results/WindowICP_EuRoC}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_SEQUENCES=(MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203)
if [[ "$#" -gt 0 ]]; then
    SEQUENCES=("$@")
else
    SEQUENCES=("${DEFAULT_SEQUENCES[@]}")
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable is unavailable: $PYTHON_BIN" >&2
    exit 1
fi

for sequence in "${SEQUENCES[@]}"; do
    config="Config/Sequence/EuRoC_${sequence}_local.yaml"
    if [[ ! -f "$config" ]]; then
        echo "Missing sequence config: $config" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT/launcher_logs"
exec 9>"$RESULT_ROOT/.window_icp_batch.lock"
if ! flock -n 9; then
    echo "Another WindowICP batch is already using $RESULT_ROOT" >&2
    exit 1
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
launcher_log="$RESULT_ROOT/launcher_logs/mh02_v203_${timestamp}.log"

command=(
    "$PYTHON_BIN"
    Scripts/Experiment/CompareWindowICP.py
    --sequences "${SEQUENCES[@]}"
    --modes window_skip
    --seed "$SEED"
    --resultRoot "$RESULT_ROOT"
)

echo "MAC-VO five-frame WindowICP batch"
echo "Sequences: ${SEQUENCES[*]}"
echo "Seed: $SEED"
echo "Results: $RESULT_ROOT"
echo "Launcher log: $launcher_log"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

set -o pipefail
"${command[@]}" 2>&1 | tee "$launcher_log"

latest_metrics="$(find "$RESULT_ROOT" -mindepth 2 -maxdepth 2 -type f -name metrics.csv \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "$latest_metrics" || ! -f "$latest_metrics" ]]; then
    echo "Batch finished without a metrics.csv file" >&2
    exit 1
fi

echo "All requested sequences completed and evaluated."
echo "Metrics CSV: $latest_metrics"
echo "Metrics JSON: ${latest_metrics%.csv}.json"
