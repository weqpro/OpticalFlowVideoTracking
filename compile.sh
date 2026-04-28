#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="Release"
RUN_TESTS=false

while getopts "dt" opt; do
    case $opt in
        d) BUILD_TYPE="Debug" ;;
        t) RUN_TESTS=true ;;
        *) echo "Usage: $0 [-d] [-t]"; exit 1 ;;
    esac
done

mkdir -p build

cmake -B build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_CLANG_TIDY=ON

cmake --build build

if [ "$RUN_TESTS" = true ]; then
    ctest --test-dir build --output-on-failure
fi
