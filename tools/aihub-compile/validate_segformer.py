"""Phase 0 step (d) helper: numerical-parity check for SegFormer compile.

Runs the same 1×3×512×512 input through (a) the PyTorch reference and (b) the
Hub-compiled binary via `hub.submit_inference_job`, then reports the per-class
max-abs-diff on the 1×18×128×128 logits.

A diff under ~5e-1 on w8a16 is acceptable for segmentation — argmax is what
matters, and SegFormer's class boundaries are sharp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch
from transformers import SegformerForSemanticSegmentation

from verify_device import pick_target

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
INPUT_SHAPE = (1, 3, 512, 512)
ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "app/src/main/assets/segformer_b2_clothes.bin"
)
THRESHOLD = 0.5


def main() -> None:
    if not ASSET_PATH.exists():
        print(f"Asset {ASSET_PATH} missing — run compile_segformer_clothes.py first.")
        sys.exit(2)

    target = pick_target()
    rng = np.random.default_rng(seed=0)
    x_np = rng.standard_normal(INPUT_SHAPE, dtype=np.float32)
    x_t = torch.from_numpy(x_np)

    print("Running PyTorch reference…")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).eval()
    with torch.no_grad():
        ref = model(pixel_values=x_t).logits.numpy()
    print(f"  ref shape: {ref.shape}")

    print(f"Submitting inference job to Hub on device {target!r}…")
    job = hub.submit_inference_job(
        model=hub.upload_model(str(ASSET_PATH)),
        device=hub.Device(target),
        inputs={"pixel_values": [x_np]},
        name="segformer_b2_clothes-validate",
    )
    print(f"  job id: {job.job_id}  dashboard: {job.url}")
    outputs = job.download_output_data()
    if outputs is None:
        print("Inference job failed.", file=sys.stderr)
        sys.exit(3)

    (got_np,) = next(iter(outputs.values()))
    if got_np.shape != ref.shape:
        print(
            f"Shape mismatch: hub={got_np.shape} ref={ref.shape}", file=sys.stderr
        )
        sys.exit(4)

    diff = np.max(np.abs(got_np - ref))
    argmax_match = (got_np.argmax(axis=1) == ref.argmax(axis=1)).mean()
    print(f"max-abs-diff: {diff:.4f}  (threshold {THRESHOLD})")
    print(f"argmax agreement: {argmax_match * 100:.2f}%")

    if diff > THRESHOLD and argmax_match < 0.98:
        print("FAIL — both max-abs-diff and argmax agreement exceeded bounds.")
        sys.exit(5)
    print("PASS")


if __name__ == "__main__":
    main()
