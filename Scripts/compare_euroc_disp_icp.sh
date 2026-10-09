#!/usr/bin/env bash
set -Eeuo pipefail

MACVO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MACVO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$MACVO_ROOT/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$MACVO_ROOT/Results}"
if [[ "$#" -eq 0 ]]; then
    SEQUENCES=(MH01)
else
    SEQUENCES=("$@")
fi

for seq in "${SEQUENCES[@]}"; do
    data_cfg="Config/Sequence/EuRoC_${seq}_local.yaml"

    for mode in disp icp; do
        if [[ "$mode" == "disp" ]]; then
            odom_cfg="Config/Experiment/MACVO/MACVO_Fast_local.yaml"
            project="MACVO-Fast"
        else
            odom_cfg="Config/Experiment/MACVO/MACVO_Fast_ICP_local.yaml"
            project="MACVO-Fast-ICP"
        fi

        if [[ "$mode" == "disp" ]] && find "$RESULT_ROOT/${project}@${seq}" \
            -mindepth 2 -maxdepth 2 -name poses.npy -print -quit 2>/dev/null | grep -q .; then
            echo "Reuse existing ${project}@${seq} result"
        else
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PYTHON_BIN" MACVO.py \
                --odom "$odom_cfg" \
                --data "$data_cfg" \
                --resultRoot "$RESULT_ROOT" \
                --noeval --timing
        fi
    done

    disp_space="$(find "$RESULT_ROOT/MACVO-Fast@${seq}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
    icp_space="$(find "$RESULT_ROOT/MACVO-Fast-ICP@${seq}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"

    "$PYTHON_BIN" -m Evaluation.EvalSeq --spaces "$disp_space" "$icp_space"
done
