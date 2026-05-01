#!/usr/bin/env bash
# push-model.sh <path-to-task-file> [debug|release]
# Copies a Gemma .task file into LocalAi's filesDir/models/ via run-as.
# Only works for debug builds (or any build where run-as is permitted).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <path-to-.task> [debug|release]" >&2
    exit 1
fi

SRC="$1"
VARIANT="${2:-debug}"
case "$VARIANT" in
    debug)   PKG="com.nothing.localai.debug" ;;
    release) PKG="com.nothing.localai" ;;
    *) echo "variant must be debug or release"; exit 1 ;;
esac

if [ ! -f "$SRC" ]; then
    echo "no such file: $SRC" >&2
    exit 1
fi

NAME="$(basename "$SRC")"
echo "→ pushing $NAME to /data/local/tmp"
adb push "$SRC" "/data/local/tmp/$NAME"

echo "→ moving into $PKG filesDir/models/"
adb shell "run-as $PKG mkdir -p files/models"
adb shell "run-as $PKG sh -c 'cp /data/local/tmp/$NAME files/models/ && ls -lh files/models/'"

echo "→ cleaning /data/local/tmp"
adb shell "rm /data/local/tmp/$NAME"

echo "✓ done. Model in $PKG filesDir/models/$NAME"
