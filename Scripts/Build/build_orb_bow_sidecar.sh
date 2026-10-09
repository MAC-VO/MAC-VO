#!/usr/bin/env bash
set -euo pipefail

MACVO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORB_SOURCE="${ORB_SLAM3_ROOT:-/home/zzh/ORB_SLAM3}"
BUILD_DIR="$MACVO_ROOT/build/orb_bow"
VOCAB_CACHE="$MACVO_ROOT/cache/ORBvoc.txt"
VOCAB_ARCHIVE="$ORB_SOURCE/Vocabulary/ORBvoc.txt.tar.gz"

test -f "$ORB_SOURCE/Thirdparty/DBoW2/DBoW2/FORB.cpp"
mkdir -p "$BUILD_DIR" "$MACVO_ROOT/cache"

if [[ ! -f "$VOCAB_CACHE" && -f "$VOCAB_ARCHIVE" ]]; then
    tar -xzf "$VOCAB_ARCHIVE" -C "$MACVO_ROOT/cache"
fi

cmake -S "$MACVO_ROOT/Tools/ORBBoW" -B "$BUILD_DIR" \
    -DORB_SLAM3_ROOT="$ORB_SOURCE" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel 2
