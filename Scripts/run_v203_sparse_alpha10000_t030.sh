#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="${WORKTREE:-$SCRIPT_ROOT}"
PYTHON="${PYTHON:-/home/zzh/MACVO/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/home/zzh/MACVO/Results/SparseORBLoop_V203_alpha10000_t030}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if (( $# )); then
    echo "Usage: $(basename "$0") [--dry-run]" >&2
    exit 2
fi

command=(
    "$PYTHON" Scripts/Experiment/CompareOnlineORBLoop.py
    --sequence V203
    --seq-from 0
    --seed 0
    --modes window_orb_loop_sparse_t030
    --result-root "$RESULT_ROOT"
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WORKTREE %s\n' "$WORKTREE"
    printf 'RESULT_ROOT %s\n' "$RESULT_ROOT"
    printf 'COMMAND'
    printf ' %s' "${command[@]}"
    printf '\n'
    exit 0
fi

required=(
    "$PYTHON"
    "$WORKTREE/Scripts/Experiment/CompareOnlineORBLoop.py"
    "$WORKTREE/Config/Experiment/MACVO/MACVO_Fast_WindowICP_ORBLoop_Sparse_t030.yaml"
    "$WORKTREE/Config/Sequence/EuRoC_V203_local.yaml"
    "$WORKTREE/build/orb_bow/macvo_orb_bow"
    "$WORKTREE/cache/ORBvoc.txt"
    "$WORKTREE/Model/MACVO_FrontendCov.pth"
)
for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "MISSING $path" >&2
        exit 2
    fi
done
if [[ ! -x "$PYTHON" || ! -x "$WORKTREE/build/orb_bow/macvo_orb_bow" ]]; then
    echo "MISSING executable Python or ORB sidecar" >&2
    exit 2
fi

cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${command[@]}"
