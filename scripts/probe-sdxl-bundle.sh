#!/usr/bin/env bash
# probe-sdxl-bundle.sh <path-to-unet.bin> [debug|release]
#
# Push a single UNet QNN context binary into LocalAi's
# filesDir/models/sd-xl-verify/ so the next app launch runs the boot-time
# SDXL compatibility probe (see LocalAiApp.probeSdxlVerifyIfPresent).
#
# Purpose: cheap one-shot check of whether a V75-compiled SDXL bundle
# (e.g. an xororz _8gen3.zip variant) instantiates on V73 silicon before
# committing to the full SDXL integration. The probe reports init +
# inspectBinary + instantiate (the latter is the call that returns
# rc=0x... on Hexagon-revision mismatch).
#
# Workflow (illustrious DMD2 4-step example):
#   curl -L -o /tmp/sdxl.zip \
#     https://huggingface.co/xororz/sdxl-qnn/resolve/main/illustrious_v16_dmd2_qnn2.28_8gen3.zip
#   unzip /tmp/sdxl.zip -d /tmp/sdxl
#   find /tmp/sdxl -name unet.bin
#   ./scripts/probe-sdxl-bundle.sh /tmp/sdxl/<.../unet.bin>
#   ./gradlew :app:installDebug
#   adb logcat -c
#   adb logcat -s LocalAiApp:V imagegen:V qnn_session:V \
#     | grep -E 'sdxl-probe|VERDICT|rc=0x'
#
# The probe runs once per app launch. Re-push to re-probe.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <path-to-unet.bin> [debug|release]" >&2
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

BYTES=$(stat -f%z "$SRC" 2>/dev/null || stat -c%s "$SRC")
echo "→ pushing UNet ($BYTES bytes) to $PKG sd-xl-verify/unet.bin"

TMP="/data/local/tmp/sdxl-verify-$$.bin"
adb push "$SRC" "$TMP" >/dev/null
adb shell "run-as $PKG mkdir -p files/models/sd-xl-verify"
adb shell "run-as $PKG cp $TMP files/models/sd-xl-verify/unet.bin"
adb shell "run-as $PKG ls -lh files/models/sd-xl-verify/"
adb shell "rm -f $TMP"

echo
echo "✓ staged. To run the probe:"
echo "    ./gradlew :app:installDebug   # or just relaunch the app"
echo "    adb logcat -c"
echo "    adb logcat -s LocalAiApp:V imagegen:V qnn_session:V | grep -E 'sdxl-probe|VERDICT|rc=0x'"
echo
echo "  PASS  → 'sdxl-probe: VERDICT: bundle loads on this device'"
echo "  FAIL  → 'sdxl-probe: instantiate: FAIL ... rc=0x...'  ← V75→V73 incompat"
