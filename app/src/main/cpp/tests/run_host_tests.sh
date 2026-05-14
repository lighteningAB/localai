#!/usr/bin/env bash
# Compile and run host-side native unit tests on macOS / Linux.
# These tests exercise the pure-C++ pieces (scheduler, tokenizer later) that
# don't depend on Android, QNN, or JNI — so they run in <1s on the dev machine.
#
# Usage:
#   ./app/src/main/cpp/tests/run_host_tests.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="${TMPDIR:-/tmp}/imagegen-host-tests"
mkdir -p "$BUILD_DIR"

CXX="${CXX:-clang++}"
FLAGS=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -I"$CPP_ROOT")

run_test() {
    local name="$1"; shift
    local out="$BUILD_DIR/$name"
    echo "==> building $name"
    "$CXX" "${FLAGS[@]}" "$@" -o "$out"
    echo "==> running $name"
    "$out"
    echo
}

run_test scheduler_test \
    "$CPP_ROOT/scheduler.cpp" \
    "$HERE/scheduler_test.cpp"

# Tokenizer test. Needs a CLIP tokenizer.json — defaults to one preserved at
# /tmp/imagegen-tokenizer.json (extracted from any xororz bundle). Set
# IMAGEGEN_TEST_TOKENIZER_JSON to override. Skipped if not present.
TOK_JSON="${IMAGEGEN_TEST_TOKENIZER_JSON:-/tmp/imagegen-tokenizer.json}"
if [ -f "$TOK_JSON" ]; then
    IMAGEGEN_TEST_TOKENIZER_JSON="$TOK_JSON" \
        run_test tokenizer_test \
            "$CPP_ROOT/tokenizer.cpp" \
            "$HERE/tokenizer_test.cpp"
else
    echo "==> SKIP tokenizer_test (no $TOK_JSON; set IMAGEGEN_TEST_TOKENIZER_JSON)"
fi

# Optional: bundle_loader test. Extracts a downloaded xororz bundle into a temp
# dir on demand so the test has real files to validate against. Skipped if no
# bundle zip is present.
BUNDLE_ZIP="${IMAGEGEN_TEST_BUNDLE_ZIP:-/tmp/AbsoluteReality_qnn2.28_8gen2.zip}"
BUNDLE_EXTRACT_DIR="${BUILD_DIR}/bundle"
if [ -f "$BUNDLE_ZIP" ]; then
    if [ ! -d "$BUNDLE_EXTRACT_DIR/output_512/qnn_models_8gen2" ]; then
        echo "==> extracting bundle for bundle_loader_test"
        mkdir -p "$BUNDLE_EXTRACT_DIR"
        unzip -q -o "$BUNDLE_ZIP" -d "$BUNDLE_EXTRACT_DIR"
    fi
    IMAGEGEN_TEST_BUNDLE_DIR="$BUNDLE_EXTRACT_DIR/output_512/qnn_models_8gen2" \
        run_test bundle_loader_test \
            "$CPP_ROOT/bundle_loader.cpp" \
            "$HERE/bundle_loader_test.cpp"
else
    echo "==> SKIP bundle_loader_test (no $BUNDLE_ZIP; set IMAGEGEN_TEST_BUNDLE_ZIP)"
fi

echo "All host tests passed."
