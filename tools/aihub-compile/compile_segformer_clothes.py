"""Phase 0 step (b): compile mattmdjaga/segformer_b2_clothes via Qualcomm AI Hub.

Output: `app/src/main/assets/segformer_b2_clothes.bin` — a QNN context binary
loadable by the existing `qnn_session` runner. Single 1×3×512×512 RGB input,
single 1×18×128×128 logits output (SegFormer's decoder upsamples once; we
upsample the rest in C++ via nearest-neighbor to keep on-device cost low).

Run after `verify_device.py`. Re-run only when:
  - The HF model card revision changes (pin via env REVISION=<sha>).
  - QAIRT major version changes on either side.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import qai_hub as hub
import torch
from transformers import SegformerForSemanticSegmentation

from verify_device import pick_target

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
INPUT_SHAPE = (1, 3, 512, 512)
ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "app/src/main/assets/segformer_b2_clothes.tflite"
)


class SegformerLogits(torch.nn.Module):
    """Wraps the HF model so trace returns the raw logits tensor."""

    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def main() -> None:
    target = pick_target()
    revision = os.environ.get("REVISION") or None
    print(f"Loading {MODEL_ID} (revision={revision or 'default'})…")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID, revision=revision
    ).eval()
    wrapped = SegformerLogits(model).eval()

    example = torch.randn(*INPUT_SHAPE)
    print(f"Tracing on input shape {INPUT_SHAPE}…")
    traced = torch.jit.trace(wrapped, example, strict=False)

    print(f"Submitting compile job to AI Hub for device {target!r}…")
    job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(target),
        input_specs={"pixel_values": (INPUT_SHAPE, "float32")},
        options=" ".join(
            [
                # TFLite target — Hub's QNN context-binary blobs all fail
                # rc=0x138d on this device's runtime. LiteRT 1.0.1 is already
                # in the APK and provides CPU + GPU delegates.
                "--target_runtime tflite",
            ]
        ),
        name="segformer_b2_clothes",
    )
    print(f"  job id: {job.job_id}")
    print(f"  dashboard: {job.url}")

    target_model = job.get_target_model()
    if target_model is None:
        print("Compile job failed — see dashboard for diagnostics.", file=sys.stderr)
        sys.exit(3)

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded = target_model.download(str(ASSET_PATH.with_suffix(".tmp")))
    # `download` returns a path; rename atomically to the final asset location.
    shutil.move(downloaded, ASSET_PATH)
    print(f"Wrote {ASSET_PATH} ({ASSET_PATH.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
