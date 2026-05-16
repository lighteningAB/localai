# aihub-compile

One-time compile + validation harness for the two new outfit-swap models. Run
on a host machine with network access to Qualcomm AI Hub; the resulting `.bin`
context binaries are committed under `app/src/main/assets/`.

## Setup

```bash
cd tools/aihub-compile
uv sync                       # or: pip install -e .
qai-hub configure --api_token $QAI_HUB_TOKEN
```

## Scripts

| Script | Output | Time |
|---|---|---|
| `verify_device.py` | prints whether `Snapdragon 8s Gen 4` is in the qai-hub Device catalog, plus the chosen compile target | <1 s |
| `compile_segformer_clothes.py` | `app/src/main/assets/segformer_b2_clothes.bin` | 10-20 min |
| `compile_sd15_inpaint_unet.py` | `app/src/main/assets/sd15_inpaint_unet.bin` | 20-40 min |
| `validate_segformer.py` | logs max-abs-diff vs PyTorch reference | 1-2 min |
| `validate_inpaint_unet.py` | logs max-abs-diff vs PyTorch reference | 1-2 min |

## Phase 0 order

1. `python verify_device.py` — confirms the compile target. Hard gate (a).
2. `python compile_segformer_clothes.py` — produces the segmenter binary.
3. Push to device, load via `bundle_loader`, dump a mask PNG. Hard gate (c)+(d).

Phase 1 then runs `compile_sd15_inpaint_unet.py` + `validate_inpaint_unet.py`.
