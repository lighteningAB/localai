"""Phase 1b: compile MI-GAN (picsart-ai-research) via Qualcomm AI Hub.

MI-GAN is a single-pass context-fill GAN that erases the masked region of a
photo. Used as Stage 1 of the outfit-swap pipeline (PLAN-OUTFIT-SWAP.md §13.2):
mask → MI-GAN erase → base UNet img2img with text prompt → composite.

Source: https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx
Inputs (ONNX, dynamic axes — fixed at compile time to 512×512):
  image: uint8 [1, 3, 512, 512]   RGB
  mask:  uint8 [1, 1, 512, 512]   255 = known, 0 = masked
Output:
  result: uint8 [1, 3, 512, 512]  RGB

Output binary: app/src/main/assets/migan.tflite (TFLite target — Hub's QNN
blobs fail rc=0x138d on this device).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import qai_hub as hub

from verify_device import pick_target

# Core model (no preprocessing pipeline). The pipeline variant fails with
# "Computed 'sizes' input for layer '.Resize' is not supported" — its dynamic
# resize ops aren't TFLite-convertible. We do resize in Kotlin instead.
ONNX_PATH = Path(__file__).parent / "migan.onnx"
ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "app/src/main/assets/migan.tflite"
)
SIZE = 512


def main() -> None:
    if not ONNX_PATH.exists():
        print(
            f"Missing {ONNX_PATH}. Download with:\n"
            "  curl -L https://huggingface.co/andraniksargsyan/migan/"
            "resolve/main/migan_pipeline_v2.onnx -o " + str(ONNX_PATH),
            file=sys.stderr,
        )
        sys.exit(2)

    target = pick_target()
    print(f"Submitting MI-GAN compile job to AI Hub for device {target!r}…")
    job = hub.submit_compile_job(
        model=str(ONNX_PATH),
        device=hub.Device(target),
        input_specs={
            "image": ((1, 3, SIZE, SIZE), "uint8"),
            "mask":  ((1, 1, SIZE, SIZE), "uint8"),
        },
        # MI-GAN's ScatterND pipeline emits int64 intermediates that the TFLite
        # converter rejects without the explicit truncation flag.
        options="--target_runtime tflite --truncate_64bit_tensors --truncate_64bit_io",
        name="migan_pipeline_v2",
    )
    print(f"  job id: {job.job_id}")
    print(f"  dashboard: {job.url}")

    tgt = job.get_target_model()
    if tgt is None:
        print("Compile job failed — see dashboard.", file=sys.stderr)
        sys.exit(3)

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded = tgt.download(str(ASSET_PATH.with_suffix(".tmp")))
    shutil.move(downloaded, ASSET_PATH)
    print(f"Wrote {ASSET_PATH} ({ASSET_PATH.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
