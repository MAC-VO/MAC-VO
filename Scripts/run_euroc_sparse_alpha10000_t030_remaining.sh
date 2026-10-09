#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="${WORKTREE:-$SCRIPT_ROOT}"
DEFAULT_SEQUENCES=(MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202)

if [[ "${1:-}" == "--list-defaults" ]]; then
    echo "${DEFAULT_SEQUENCES[*]}"
    exit 0
fi

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0") [--dry-run] [SEQUENCE ...]"
    echo "Defaults: ${DEFAULT_SEQUENCES[*]}"
    exit 0
fi

export WORKTREE
export RESULT_ROOT="${RESULT_ROOT:-/home/zzh/MACVO/Results/SparseORBLoop_EuRoC_alpha10000_t030}"
export MODE="window_orb_loop_sparse_t030"
export ODOM_CONFIG="$WORKTREE/Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml"

BATCH_SCRIPT="$SCRIPT_ROOT/Scripts/run_euroc_sparse_alpha10000_remaining.sh"
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    if (( $# )); then
        exec "$BATCH_SCRIPT" --dry-run "$@"
    fi
    exec "$BATCH_SCRIPT" --dry-run "${DEFAULT_SEQUENCES[@]}"
fi

if (( $# )); then
    exec "$BATCH_SCRIPT" "$@"
fi
exec "$BATCH_SCRIPT" "${DEFAULT_SEQUENCES[@]}"
