"""Phase 1 gate: numerical-parity check for the SD 1.5 inpaint UNet compile.

w8a16-quantized UNet's noise prediction will drift from float reference, but
the drift must stay bounded; otherwise diffusion accumulates error and image
quality collapses by step ~10/20. We compare a single forward pass at t=500.

Acceptance: max-abs-diff on the noise prediction under ~0.2. (Empirically the
existing base UNet sits around 0.05-0.1 at this quant level.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch
from diffusers import UNet2DConditionModel

from verify_device import pick_target

MODEL_ID = "runwayml/stable-diffusion-inpainting"
SUBFOLDER = "unet"
LATENT_SHAPE = (1, 9, 64, 64)
TEXT_EMB_SHAPE = (1, 77, 768)
ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "app/src/main/assets/sd15_inpaint_unet.bin"
)
THRESHOLD = 0.20


def main() -> None:
    if not ASSET_PATH.exists():
        print(f"Asset {ASSET_PATH} missing — run compile_sd15_inpaint_unet.py first.")
        sys.exit(2)

    target = pick_target()
    rng = np.random.default_rng(seed=0)
    sample_np = rng.standard_normal(LATENT_SHAPE, dtype=np.float32)
    text_np = rng.standard_normal(TEXT_EMB_SHAPE, dtype=np.float32)
    ts_np = np.array([500], dtype=np.int64)

    print("Running PyTorch reference…")
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder=SUBFOLDER, torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        ref = unet(
            sample=torch.from_numpy(sample_np),
            timestep=torch.from_numpy(ts_np),
            encoder_hidden_states=torch.from_numpy(text_np),
            return_dict=False,
        )[0].numpy()
    print(f"  ref shape: {ref.shape}")

    print(f"Submitting inference job to Hub on device {target!r}…")
    job = hub.submit_inference_job(
        model=hub.upload_model(str(ASSET_PATH)),
        device=hub.Device(target),
        inputs={
            "sample": [sample_np],
            "timestep": [ts_np],
            "encoder_hidden_states": [text_np],
        },
        name="sd15_inpaint_unet-validate",
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
    rms = float(np.sqrt(np.mean((got_np - ref) ** 2)))
    print(f"max-abs-diff: {diff:.4f}  (threshold {THRESHOLD})")
    print(f"rms-diff:     {rms:.4f}")

    if diff > THRESHOLD:
        print("FAIL — max-abs-diff exceeded threshold.")
        sys.exit(5)
    print("PASS")


if __name__ == "__main__":
    main()
