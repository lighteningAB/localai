"""Phase 1: compile SD 1.5 inpaint UNet via Qualcomm AI Hub.

Differs from the base SD 1.5 UNet only in the first convolution: input channels
are 9 (noisy latent 4 + mask 1 + masked-image latent 4) instead of 4. Everything
downstream is identical.

Output: `app/src/main/assets/sd15_inpaint_unet.bin`.

Run after Phase 0 lands. Re-run only on model revision change or QAIRT bump.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import qai_hub as hub
import torch
from diffusers import UNet2DConditionModel

from verify_device import pick_target

MODEL_ID = "runwayml/stable-diffusion-inpainting"
SUBFOLDER = "unet"
LATENT_SHAPE = (1, 9, 64, 64)
TIMESTEP_SHAPE = (1,)
TEXT_EMB_SHAPE = (1, 77, 768)
ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "app/src/main/assets/sd15_inpaint_unet.tflite"
)


class InpaintUnetForward(torch.nn.Module):
    """Trace-friendly wrapper: positional args, raw tensor output."""

    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def main() -> None:
    target = pick_target()
    revision = os.environ.get("REVISION") or None
    print(f"Loading {MODEL_ID}/{SUBFOLDER} (revision={revision or 'default'})…")
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder=SUBFOLDER, revision=revision, torch_dtype=torch.float32
    ).eval()
    wrapped = InpaintUnetForward(unet).eval()

    example = (
        torch.randn(*LATENT_SHAPE),
        torch.tensor([500], dtype=torch.long),
        torch.randn(*TEXT_EMB_SHAPE),
    )
    print(
        f"Tracing on shapes sample={LATENT_SHAPE} "
        f"timestep={TIMESTEP_SHAPE} text_emb={TEXT_EMB_SHAPE}…"
    )
    traced = torch.jit.trace(wrapped, example, strict=False)

    print(f"Submitting compile job to AI Hub for device {target!r}…")
    job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(target),
        input_specs={
            "sample": (LATENT_SHAPE, "float32"),
            "timestep": (TIMESTEP_SHAPE, "int64"),
            "encoder_hidden_states": (TEXT_EMB_SHAPE, "float32"),
        },
        options=" ".join(
            [
                # TFLite target — Hub's QNN blobs fail on this device's runtime
                # (rc=0x138d). LiteRT 1.0.1 is in the APK with XNNPACK CPU.
                "--target_runtime tflite",
                # w8a16 keeps the file size down (~830 MB vs ~3.3 GB float).
                # CPU+XNNPACK accepts int8 weights with float activations.
                "--quantize_full_type w8a16",
                # Diffusers passes the timestep as int64; truncate for TFLite.
                "--truncate_64bit_io",
            ]
        ),
        name="sd15_inpaint_unet",
    )
    print(f"  job id: {job.job_id}")
    print(f"  dashboard: {job.url}")

    target_model = job.get_target_model()
    if target_model is None:
        print("Compile job failed — see dashboard for diagnostics.", file=sys.stderr)
        sys.exit(3)

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded = target_model.download(str(ASSET_PATH.with_suffix(".tmp")))
    shutil.move(downloaded, ASSET_PATH)
    print(f"Wrote {ASSET_PATH} ({ASSET_PATH.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
