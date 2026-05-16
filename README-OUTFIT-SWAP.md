# Outfit-Swap — What We Are Adding

Companion to [`PLAN-OUTFIT-SWAP.md`](./PLAN-OUTFIT-SWAP.md). This file enumerates the concrete additions needed to ship the `outfit-swap-1` widget. Anything not on this list does not change.

**Target:** Snapdragon 8s Gen 4 (Nothing Phone 3 class) · QAIRT 2.46 · on-device only.

---

## Status (2026-05-16) — v1.1 shipped

The pipeline shipped, but the path diverged from the plan below because the original Stable Diffusion **inpaint** UNet would not compile to a loadable QNN binary on this device (Hub `rc=0x138d`). v1.1 uses the existing **base** SD 1.5 UNet via an img2img path with RePaint-style latent blending — see [`PLAN-IMG2IMG.md`](./PLAN-IMG2IMG.md) for the design details.

**Pipeline as it runs today:**

```
input photo (any size, JPEG/PNG)
   │ center-crop + resize
   ▼
512×512 RGB Bitmap
   │
   ├─► SegFormer-B2-Clothes (LiteRT, ~2.5s) → 18-class garment classmap
   │   │
   │   ▼
   │   classmap → binary garment mask → dilate 5px → feather σ=4 (composite) / σ=2 (latent)
   │
   ├─► MI-GAN context-fill (LiteRT, ~1.6s) → erased portrait (skin/bg in garment region)
   │
   ▼
SD 1.5 img2img (QNN base UNet, ~6.4s loop @ iters=12, strength=0.7)
   │ z₀ = VAE-encode(erased)                        ← QNN vae_encoder.bin
   │ z_t = √ᾱ_t · z₀ + √(1−ᾱ_t) · ε                  ← noise init at t = strength·999
   │ for i in [kStart..iters):
   │     ε̂_cond, ε̂_uncond = UNet(z, t, text)         ← QNN unet.bin (xororz)
   │     z = scheduler.step(CFG(ε̂), i, z)
   │     z = mask · z + (1−mask) · noised(z₀, t_next) ← RePaint latent blending
   │
   ▼
VAE-decode (QNN vae_decoder.bin, ~0.8s) → 512×512 RGB
   │
   ▼
feathered alpha composite (Kotlin):
   outside SegFormer mask → original photo (bit-exact face/hair/bg)
   inside  SegFormer mask → img2img output (new garment)
   │
   ▼
final 512×512 PNG via PFD pipe
```

End-to-end probe time on Nothing Phone 3: **~26s** (seg 2.5 + migan 1.6 + img2img 15 + composite 0.5 + I/O ≈ 6s). The `OutfitSwapProbe` boot path writes the final PNG to `filesDir/segformer-probe/outfit-swap-debug.png` for visual verification.

**What's different from the original plan:**
- ❌ SD 1.5 inpaint UNet — abandoned (Hub couldn't compile a loadable artifact).
- ✅ Base SD 1.5 UNet (xororz bundle) — already on-device; reused via img2img.
- ✅ SegFormer-B2-Clothes — same as planned, shipping as `segformer_b2_clothes.tflite` (LiteRT, not QNN).
- ➕ MI-GAN — new addition for context-fill (LiteRT `migan.tflite`). Not in the original plan; gives img2img a clean body silhouette to paint over.
- ➕ RePaint latent blending — new addition. At every diffusion step, the outside-mask region of the latent is replaced with a re-noised copy of z₀ at the next timestep's noise level, broadcast over the four latent channels. Eliminates anatomy drift (extra hands, wrong leg shape) and boundary seams that base img2img alone produced.

**Known residual quality gap vs. Google Doppl / TryOnDiffusion:**
- Text-conditioned, not garment-image-conditioned (no IP-Adapter yet).
- No pose-skeleton conditioning (no ControlNet-OpenPose yet).
- VAE round-trip blur inside the mask is visible against the sharp original-photo pixels outside.

**Recommended next-step interventions** (any order, all on-device):
1. **Poisson seamless cloning** in `compositeAlphaSimple` to eliminate the VAE-roundtrip seam — preserves img2img gradients, locks boundary values to the original. ~150 lines of Kotlin; sub-second on 512×512.
2. **Cheaper color match** as a Poisson alternative — sample original RGB mean in a 10px ring just outside the mask, sample img2img mean just inside, shift the entire generated region by the per-channel delta before compositing.
3. **IP-Adapter** to add garment-image conditioning to the existing UNet. ~50MB extra weights, modifies cross-attention path. Highest-leverage single addition for quality.
4. **ControlNet-OpenPose** to lock anatomy via explicit skeleton conditioning. Heavier (~700MB ControlNet + a pose-estimator model).

---

## 1. New on-device models (compiled via Qualcomm AI Hub)

| Model | Source | Approx. size | Role |
|---|---|---|---|
| **SegFormer-B2-Clothes** | `mattmdjaga/segformer_b2_clothes` | ~30-60 MB w8a16 | 18-class garment segmentation → binary mask |
| **SD 1.5 inpaint UNet** | `runwayml/stable-diffusion-inpainting` (unet) | ~1.5 GB w8a16 | 9-channel inpaint diffusion; replaces base UNet |

Compiled artifacts land under `app/src/main/assets/` as `segformer_b2_clothes.bin` and `sd15_inpaint_unet.bin`.

---

## 2. Existing model wired up

- **VAE encoder** — `vae_encoder.bin` already ships in the xororz bundle and is already required by `ImageGenRunner.isReady()`, but never loaded. Adds one `loadVaeEncoder()` mirroring `loadVaeDecoder()`.

---

## 3. New compile tooling

Lives under `tools/aihub-compile/` (new directory).

- `pyproject.toml` — `qai-hub`, `torch`, `transformers`, `diffusers`.
- `compile_segformer_clothes.py` — trace mattmdjaga model, submit Hub compile job, download `.bin`.
- `compile_sd15_inpaint_unet.py` — trace runwayml inpaint UNet (9-ch input), submit, download `.bin`.
- `validate_segformer.py`, `validate_inpaint_unet.py` — numerical-parity vs PyTorch reference.

Run once per architecture revision. Each Hub compile job takes ~10-30 minutes on the farm.

---

## 4. Native (C++) additions

All paths under `app/src/main/cpp/`.

- **`mask_ops.cpp` / `mask_ops.hpp`** (new) — classmap → binary mask → dilate (5 px) → Gaussian feather (σ=2) → 8× nearest-neighbor downsample to latent space.
- **`diffusion.cpp` / `diffusion.hpp`** — three new loaders (`loadVaeEncoder`, `loadSegFormerClothes`, `loadInpaintUnet`) and a new entry point:
  ```cpp
  Result runOutfitSwap(
      const uint8_t* png_bytes, size_t png_len,
      const std::string& prompt,
      uint32_t garment_mask_classes,
      int iterations, uint64_t seed,
      StepCallback on_step,
      ResultCallback on_result);
  ```
  The diffusion loop builds the UNet input as `concat(z_t, mask_latent, masked_img_latent)` (1×9×64×64), with mask/masked latents precomputed once per call.
- **`imagegen.cpp`** — one new JNI export: `Java_..._ImageGenNative_nativeRunOutfitSwap`.
- **`CMakeLists.txt`** — register `mask_ops.cpp`.

---

## 5. AIDL surface (append-only, mirrored in both repos)

File: `app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl`
```aidl
String generateOutfitSwap(
    in ParcelFileDescriptor inputPng,
    String prompt,
    String garmentSpec,       // "upper-clothes" | "skirt,pants" | "dress" |
                              // "upper-clothes,skirt,pants,dress" | "auto"
    int iterations,
    long seed,
    IImageGenCallback cb);

void cancelOutfitSwap(String requestId);
```

File: `app/src/main/aidl/com/nothing/localai/IImageGenCallback.aidl`
```aidl
void onStage(String requestId, String stageName);  // "segmenting" | "encoding" | "diffusing" | "decoding"
```

Mirrored verbatim in `Aiwidget/android/app/src/main/aidl/...`.

---

## 6. Kotlin service layer

- **`OutfitSwapRunner.kt`** (new) — coroutine lifecycle, PFD plumbing, callback marshalling. Mirrors `ImageGenRunner.kt`.
- **`LocalAiService.kt`** — one new method `generateOutfitSwap(...)` delegating to the runner; static `garmentSpec` → class-bitfield map.
- **`ImageGenNative.kt`** — one new external fn declaration `nativeRunOutfitSwap`.

---

## 7. Aiwidget additions

- **`GalleryPickerActivity.kt`** (new, `android/app/src/main/java/com/nothing/aiwidget/localai/capture/`) — launches `ACTION_OPEN_DOCUMENT` (`image/*`), copies the picked image into `cacheDir/images/picked-<uuid>.jpg`, then reuses the existing **`CapturePromiseRegistry`** to deliver the path back to JS via `resolvePath(requestId, path, "image/jpeg")` — same pattern `PhotoCaptureActivity` uses for camera capture. Declared `not-exported` in the manifest. No new permissions.

  *(Audit 2026-05-15: confirmed no existing gallery picker. `PhotoCaptureActivity` is camera-only via `ActivityResultContracts.TakePicture()`; the promise/registry plumbing is reusable but the activity itself is not.)*

- **`LocalAiBridge.kt`** — two Promise methods: `pickPhoto()` (mirrors the existing `capturePhoto()` shape, just launches the new activity) and `generateOutfitSwap(...)`. New `stage` event subtype on the existing `LocalAi.image` channel.
- **`testwidgets/outfit-swap-1/`** (new widget):
  - `widget.json` — portrait, name "Outfit Swap".
  - `src/index.tsx` — state machine `idle → picking → preview → segmenting → ready_to_generate → generating → done`. UI: photo thumb, garment chip row (Auto / Top / Bottom / Dress / Full body), prompt input, progress text, result image, save/share/try-again.
  - `src/styles.ts`.

---

## 8. Net change footprint

- **New files:** ~13 (mask_ops × 2, OutfitSwapRunner, GalleryPickerActivity, outfit-swap-1 widget × 3, compile tooling × 5).
- **Edited files:** ~8 (two AIDL files, two C++ files, LocalAiService.kt, ImageGenNative.kt, LocalAiBridge.kt, CMakeLists.txt, AndroidManifest.xml).
- **New shipped binaries:** 2 (SegFormer, inpaint UNet).
- **New runtime permissions:** 0 (`ACTION_OPEN_DOCUMENT` needs none).
- **Existing AIDL methods touched:** 0 (append-only).
- **Existing widgets touched:** 0.

---

## 9. Phase gates (from PLAN §10)

| # | Gate | Est. |
|---|---|---|
| 0 | Verify `Snapdragon 8s Gen 4` in qai-hub Device enum; compile SegFormer via Hub; load `.bin` under QAIRT 2.46 runtime; dump a mask PNG on-device. **Hard gate.** | 3-5 d |
| 1 | Compile SD 1.5 inpaint UNet via Hub; numerical parity check vs PyTorch reference. | 1-2 wk |
| 2 | Native pipeline end-to-end with a hardcoded mask (skip segmentation). | 1-2 wk |
| 3 | Wire SegFormer + mask_ops; garment-class selector logic. | 1 wk |
| 4 | AIDL surface; service wiring; bridge methods. | 3-5 d |
| 5 | `GalleryPickerActivity` + `outfit-swap-1` widget. | 1 wk |
| 6 | Quality tuning (dilation, feather, prompt template, edge-blend). | 2 wk |

**Total: 7-10 weeks at enterprise pace.**
