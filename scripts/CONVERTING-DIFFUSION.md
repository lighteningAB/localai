# On-device image generation — Qualcomm AI Hub SD models

We use Qualcomm AI Hub's pre-converted Stable Diffusion v1.5 models, driven via
LiteRT (TFLite). `ImageGenRunner` orchestrates the diffusion loop directly — no
MediaPipe Image Generator wrapper. See `KNOWN-ISSUES.md` for why MediaPipe was
abandoned for this device.

## 1. Download the AI Hub bundle

Qualcomm AI Hub hosts SD v1.5 quantized for Snapdragon:

  https://aihub.qualcomm.com/models/stable_diffusion_v1_5_quantized

Download the **TFLite** variant. You should end up with:
- `text_encoder.tflite` (~250 MB) — CLIP ViT-L/14
- `unet.tflite` (~850 MB) — denoising U-Net
- `vae_decoder.tflite` (~200 MB) — latent → RGB
- `tokenizer.json` — CLIP BPE vocab + merges (or `vocab.json` + `merges.txt`)
- `scheduler.json` — DDIM/PNDM scheduler config (alphas_cumprod, etc.)

AI Hub may distribute these in a single archive — extract before pushing. Total
~1.3 GB on disk.

## 2. Push to the device

```bash
./scripts/push-diffusion-bundle.sh /path/to/sd-aihub-bundle
```

The bundle lands at `filesDir/models/sd-v15-aihub/` (the default in
`ImageGenRunner.DEFAULT_DIFFUSION_DIR_NAME`).

## 3. Sanity check

```bash
adb shell run-as com.nothing.localai.debug ls -lh files/models/sd-v15-aihub/
```

You should see all four required `.tflite` files plus the tokenizer/scheduler
JSON. `ImageGenRunner.isReady()` checks the three `.tflite` files exist.

## 4. Generation status

**Currently the inference loop is unimplemented.** Calling `generateImage` will
return `NOT_IMPLEMENTED` even with the bundle in place. The remaining work
(documented in `ImageGenRunner.kt`'s class kdoc):

1. CLIP BPE tokenizer (Kotlin port)
2. TextEncoder.tflite forward pass via `org.tensorflow.lite.Interpreter`
3. DDIM scheduler step function
4. UNet.tflite loop with classifier-free guidance
5. VAEDecoder.tflite → Bitmap

The AIDL surface, RN bridge, and `image-gen-1` widget are wired end-to-end so
once those 5 pieces land, you'll have generation without any plumbing changes.
