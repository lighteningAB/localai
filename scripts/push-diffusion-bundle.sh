#!/usr/bin/env bash
# push-diffusion-bundle.sh <path-to-bundle-dir> [debug|release] [dest-name]
#
# Atomically copies an extracted Stable Diffusion bundle (directory of model
# artifacts — QNN context binaries + an MNN-format CLIP text encoder, e.g.
# the contents of an unzipped xororz SD-QNN HF release) into LocalAi's
# filesDir/models/<dest-name>/.
#
# Workflow:
#   curl -L -o /tmp/bundle.zip https://huggingface.co/xororz/sd-qnn/resolve/main/<name>.zip
#   unzip /tmp/bundle.zip -d /tmp/bundle
#   ./scripts/push-diffusion-bundle.sh /tmp/bundle/output_512/qnn_models_8gen2
#
# ImageGenRunner.DEFAULT_DIFFUSION_DIR_NAME is "sd-v15-xororz" — keep these
# in sync. Pass a third arg to override (e.g. "sd-v15-xororz-min" when
# A/B-ing the fallback variant).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <bundle-dir> [debug|release] [dest-name]" >&2
    echo "  bundle-dir: directory of model artifacts to push" >&2
    echo "  dest-name:  name under filesDir/models/ (default: sd-v15-xororz)" >&2
    exit 1
fi

SRC="$1"
VARIANT="${2:-debug}"
DEST_NAME="${3:-sd-v15-xororz}"

case "$VARIANT" in
    debug)   PKG="com.nothing.localai.debug" ;;
    release) PKG="com.nothing.localai" ;;
    *) echo "variant must be debug or release"; exit 1 ;;
esac

if [ ! -d "$SRC" ]; then
    echo "no such directory: $SRC" >&2
    exit 1
fi

# A bundle with zero files is almost certainly a misconfigured input.
FILE_COUNT=$(find "$SRC" -type f | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "no files in $SRC — converter did not emit anything" >&2
    exit 1
fi

TMP_PARENT="/data/local/tmp/diffusion-staging-$$"
echo "→ staging $FILE_COUNT files via $TMP_PARENT"
adb shell "mkdir -p $TMP_PARENT"

# adb push <dir> <dir> copies the source dir's *contents* into the destination,
# so we make the destination match the desired dest-name and push into it.
adb shell "mkdir -p $TMP_PARENT/$DEST_NAME"
adb push "$SRC/." "$TMP_PARENT/$DEST_NAME/" >/dev/null

echo "→ moving into $PKG filesDir/models/$DEST_NAME"
adb shell "run-as $PKG mkdir -p files/models"
# Wipe any prior bundle first so we don't end up with a half-merged directory
# from a previous push that had different files.
adb shell "run-as $PKG rm -rf files/models/$DEST_NAME"
adb shell "run-as $PKG cp -R $TMP_PARENT/$DEST_NAME files/models/$DEST_NAME"
adb shell "run-as $PKG ls -lh files/models/$DEST_NAME"

echo "→ cleaning $TMP_PARENT"
adb shell "rm -rf $TMP_PARENT"

echo "✓ done. Bundle at $PKG filesDir/models/$DEST_NAME ($FILE_COUNT files)"
