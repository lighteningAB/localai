# PLAN — Outfit-Swap Widget (localai × aiwidget)

**Status:** Draft for Phase 0 kickoff
**Owner:** patrick.fan@nothing.tech
**Target hardware:** Snapdragon 8s Gen 4 (Nothing Phone 3 class)
**Pipeline target:** On-device, identity-preserving outfit edit driven by a free-text prompt.

---

## 1. Goal & non-goals

**Goal.** A widget in aiwidget that lets a user (a) pick a photo from the gallery, (b) describe a target outfit in plain text, (c) optionally choose which garment region to edit (top / bottom / dress / auto), and (d) receive a generated image with the outfit replaced. The person's face, hair, body proportions, pose, and background must be visibly preserved.

**Non-goals for v1.**
- Garment-image-driven try-on (uploading a reference garment photo). Pure text drives v1.
- Multi-person photos. v1 targets single-subject portraits.
- Video. Still images only.
- Cloud fallback. The widget is on-device exclusive.
- SDXL or Flux backbones. v1 ships on SD 1.5 inpaint to leverage the existing localai pipeline.

---

## 2. User-facing flow

1. User opens the **outfit-swap-1** widget on the home screen.
2. Tap "Pick photo" → system document picker (`ACTION_OPEN_DOCUMENT`) → JPEG/PNG copied into `cacheDir/images/picked-<uuid>.jpg`.
3. Widget shows the thumbnail. Optional "Preview mask" button overlays the segmentation result so the user sees which region will change before paying the inference cost.
4. Garment selector chip row: `Auto · Top · Bottom · Dress · Full body`.
5. Prompt input: free text ("a black leather biker jacket", "a red summer dress").
6. Tap **Generate** → progress stages: *segmenting → encoding → diffusing (step N/T) → decoding* → result image renders in-widget.
7. Buttons: **Save to gallery · Share · Try again**.

---

## 3. Architecture

```
[Gallery photo, 512×512 RGB]
        │
        ▼
SegFormer-B2-Clothes  ──►  class map (1×18×512×512 logits → 512×512 argmax)
        │
        ▼
select classes per garment selector
(Top → {4}, Bottom → {5,6}, Dress → {7}, Full body → {4,5,6,7},
 Auto → union of any class ≠ background/face/hair/limbs/bag/scarf)
        │
        ▼
dilate (5 px) + Gaussian feather (σ=2) ──► binary mask 512×512
        │
        ▼
downsample 8× (nearest)                  ──► mask_latent (1×1×64×64)

[Gallery photo]
        │
        ▼
VAE encoder (existing asset, currently unloaded) ──► img_latent (1×4×64×64)
        │
        ▼
img_latent ⊙ (1 - mask_latent)           ──► masked_img_latent (1×4×64×64)

concat([z_t, mask_latent, masked_img_latent])    ──► UNet input (1×9×64×64)

[Prompt + CLIP text encoder (existing)] ──► text embed (1×77×768)
                                                      │
                                                      ▼
                            SD 1.5 INPAINT UNet (Hub-compiled by us)
                                                      │
                                          DDIM loop, T=20-30 steps
                                                      │
                                                      ▼
                                                  denoised z_0
                                                      │
                                                      ▼
                                          VAE decoder (existing) ──► PNG
```

Four on-device models on Hexagon: CLIP text encoder + VAE enc/dec + SegFormer + SD inpaint UNet. CLIP and VAE-dec are already in the localai pipeline. VAE-enc asset is shipped but never loaded. SegFormer and inpaint UNet are new compiles.

---

## 4. Model inventory

| Component | Source | Status | Notes |
|---|---|---|---|
| CLIP text encoder (ViT-L/14) | xororz QNN bundle | **In localai** | No change |
| SD 1.5 UNet (base, 4-channel input) | xororz QNN bundle | **In localai** | **Replaced** by inpaint UNet (see decision log §13.3) |
| SD 1.5 inpaint UNet (9-channel input) | runwayml/stable-diffusion-inpainting | **New — we compile** | Phase 1 |
| VAE encoder | xororz bundle | **Shipped, unloaded** | Wire up in Phase 2 |
| VAE decoder | xororz bundle | **In localai** | No change |
| SegFormer-B2-Clothes (18 cls, ATR) | mattmdjaga/segformer_b2_clothes | **New — we compile** | Phase 0/3 |
| **Fallback:** U2Net cloth-segmentation (4 cls) | levindabhi/cloth-segmentation | **Standby** | Activate only if SegFormer compile fails or is too slow |
| **Optional Stage-1 erase** (Phase 6 fallback only) | picsart-ai-research/mi-gan (`migan_pipeline_v2.onnx`, 512×512, MIT) | **Standby** | Activate only if §13.2 revisit-if triggers. Order-of-magnitude smaller/faster than LaMa; ONNX → Hub compiles directly. |

---

## 5. Target hardware & runtime

- **Chipset:** Snapdragon 8s Gen 4. The qai-hub SDK Device enum for 8s Gen 4 must be verified in Phase 0; if absent, fall back to `Snapdragon 8 Gen 3` as the compile target and rely on Hexagon forward-compat (QAIRT 2.x within same major).
- **Runtime:** QAIRT (localai currently uses 2.46). Hub-published binaries today are QAIRT 2.45. **Phase 0 gate** verifies 2.45 → 2.46 forward-compat empirically.
- **Quantization:** SD components stay at w8a16 to match the existing pipeline. SegFormer compiled w8a16 if Hub auto-quant supports it; else float and accept the perf hit (still <100 ms on V79-class HTP).
- **Context residency:** Cannot co-resident all four binaries. Orchestrate as: load SegFormer → run → unload → load CLIP+VAE-enc → run → unload encoders → load UNet → diffuse → unload UNet → load VAE-dec → run. Same pattern the existing pipeline uses.

---

## 6. Native pipeline changes

All paths relative to `app/src/main/cpp/`.

### `diffusion.cpp` / `diffusion.hpp`
- Add `loadVaeEncoder()` — the `.bin` asset exists but no loader path. Mirror `loadVaeDecoder()`.
- Add `loadSegFormerClothes()` — new QNN context binary load + tensor binding.
- Add `loadInpaintUnet()` — replaces the existing `loadUnet()` call site by default (see §13.3).
- New entry point `runOutfitSwap(...)`:
  ```cpp
  Result runOutfitSwap(
      const uint8_t* png_bytes, size_t png_len,
      const std::string& prompt,
      uint32_t garment_mask_classes,    // bitfield over 18 SegFormer classes
      int iterations, uint64_t seed,
      StepCallback on_step,
      ResultCallback on_result);
  ```
- New helpers in a new `mask_ops.cpp`:
  - `decodePngToRgbFloat(...)` — already exists for output; mirror for input.
  - `classmapToBinaryMask(uint8_t* classmap, uint32_t selected_classes, …)`.
  - `dilateMask(…)` — 5 px kernel, separable.
  - `featherMask(…)` — Gaussian σ=2.
  - `downsampleMaskNearest8x(…)`.
- Diffusion loop change: input tensor is now `1×9×64×64`. Build it as `concat(z_t, mask_latent_broadcast_to_1ch, masked_img_latent_4ch)` once per step (mask_latent and masked_img_latent are constant across timesteps; precompute once).

### `imagegen.cpp` (JNI)
- New JNI method `Java_..._ImageGenNative_nativeRunOutfitSwap(...)` paralleling `nativeGenerateImage`. Takes an FD for input PNG, prompt, classes bitfield, iters, seed, callback object. Wires native callbacks back through JNIEnv.

### Build / CMake
- New `mask_ops.cpp` added to `CMakeLists.txt`.
- New shipped assets: `assets/segformer_b2_clothes.bin`, `assets/sd15_inpaint_unet.bin`. Existing `assets/vae_encoder.bin` becomes referenced.

---

## 7. AIDL contract changes

File: `app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl` — **append-only** per the header comment.

```aidl
// Append at end:
String generateOutfitSwap(
    in ParcelFileDescriptor inputPng,
    String prompt,
    String garmentSpec,       // CSV of class names: "upper-clothes", "skirt,pants", "dress"
                              //                     "upper-clothes,skirt,pants,dress" for "full body"
                              //                     "auto" for heuristic union
    int iterations,
    long seed,
    IImageGenCallback cb);

void cancelOutfitSwap(String requestId);
```

**Reuse existing `IImageGenCallback`** (`onStep`, `onResult`, `onError`). Stage labels ("segmenting", "encoding", "diffusing", "decoding") are encoded into the existing `step/totalSteps` semantics: stages 1-3 are reported as `step=0, totalSteps=T+3` style negative-index markers, OR we add a single new callback method:

```aidl
// In IImageGenCallback.aidl, append:
void onStage(String requestId, String stageName);
```

Append-only on the callback interface too. Server-side: `LocalAiService.kt:183-215` pattern is mirrored in a new `LocalAiService.generateOutfitSwap()` method that delegates to a new `OutfitSwapRunner.kt`. `OutfitSwapRunner.kt` mirrors `ImageGenRunner.kt:111`-style structure: PFD plumbing, callback marshalling, lifecycle.

`garmentSpec` parsing is done service-side: a static map `{"upper-clothes": 4, "skirt": 5, "pants": 6, "dress": 7, …}` produces the classes bitfield passed to native. Unknown spec strings → error code back to widget.

---

## 8. Aiwidget changes

### New activity: gallery picker
File: `android/app/src/main/java/com/nothing/aiwidget/localai/capture/GalleryPickerActivity.kt`.
Mirror `PhotoCaptureActivity.kt` but:
- Launch `Intent(ACTION_OPEN_DOCUMENT)` with `setType("image/*")`. No `READ_MEDIA_IMAGES` permission needed under the document-picker model.
- On result: copy the content URI into `cacheDir/images/picked-<uuid>.jpg`, return the local path.
- Declare in `AndroidManifest.xml` not-exported, mirroring `PhotoCaptureActivity` (lines 166-175 region).

### Bridge extension
File: `android/app/src/main/java/com/nothing/aiwidget/localai/LocalAiBridge.kt`.
- `pickPhoto(): Promise<String>` — launches `GalleryPickerActivity`, resolves to file path.
- `generateOutfitSwap(photoPath: String, prompt: String, garmentSpec: String, iters: Int, seed: Long): Promise<String /* requestId */>` — opens the picked file as `ParcelFileDescriptor.MODE_READ_ONLY`, calls AIDL, returns requestId.
- Output streams via existing `LocalAi.image` NativeEventEmitter channel (reuse the contract that `generateImage` already established). New event subtype `stage` for stage labels.

### New widget
Directory: `testwidgets/outfit-swap-1/`.
- `widget.json`: `widget_size: "portrait"`, `name: "Outfit Swap"`, `description: "Re-dress a photo from your gallery"`, `tags: ["image", "fashion", "edit"]`, `source_dir: "src"`.
- `src/index.tsx`:
  - State machine: `idle → picking → preview → segmenting → ready_to_generate → generating → done`.
  - UI: photo thumb, garment chip row (Auto/Top/Bottom/Dress/Full body), prompt TextInput, Generate button, progress text, result Image, action buttons (Save / Share / Try again).
  - Calls `LocalAi.pickPhoto()`, optionally a future `LocalAi.previewMask()` (cheap segmentation-only call), then `LocalAi.generateOutfitSwap()`.
- No new permissions in the host manifest (the existing `READ_MEDIA_IMAGES` declaration at line 31 is not even needed for `ACTION_OPEN_DOCUMENT`, but is harmless).

---

## 9. AI Hub compile workflow

One-time setup. Lives in a new directory `tools/aihub-compile/` in localai.

- `pyproject.toml` with `qai-hub`, `torch`, `transformers`, `diffusers`.
- One script per target model:
  - `compile_segformer_clothes.py`:
    1. Load `mattmdjaga/segformer_b2_clothes` via `transformers`.
    2. Wrap in a thin module that returns logits-only (drop the HF output wrapper).
    3. `torch.jit.trace` with example input `[1, 3, 512, 512]`.
    4. `qai_hub.submit_compile_job(model=traced, device=Device("Snapdragon 8s Gen 4"), input_specs={"pixel_values": ((1,3,512,512), "float32")}, options="--target_runtime qnn_context_binary")`.
    5. Wait, download `.bin`, place under `app/src/main/assets/segformer_b2_clothes.bin`.
  - `compile_sd15_inpaint_unet.py`:
    1. Load `runwayml/stable-diffusion-inpainting` UNet via `diffusers.UNet2DConditionModel.from_pretrained(..., subfolder="unet")`.
    2. Wrap to accept inputs `(z_t [1,9,64,64], timestep [1], text_emb [1,77,768])`.
    3. Trace; submit with input specs above.
    4. Wait, download `.bin`, place under `app/src/main/assets/sd15_inpaint_unet.bin`.
- Numerical-parity check harness (`validate_segformer.py`, `validate_inpaint_unet.py`): run PyTorch reference and Hub-compiled binary on the same input via `qai_hub.submit_inference_job`, assert max-abs-diff under threshold.

Compile cost: each job ~10-30 min on Hub's farm. Run once per architecture revision.

---

## 10. Phase plan

| # | Phase | Deliverable | Est. | Gate |
|---|---|---|---|---|
| **0** | (a) Verify `Snapdragon 8s Gen 4` is a valid qai-hub `Device`; (b) compile mattmdjaga SegFormer-B2-Clothes via Hub; (c) load that `.bin` in localai's existing QNN runner and verify QAIRT 2.45 binary loads under our 2.46 runtime; (d) run inference and dump a mask PNG. | Working SegFormer mask on-device | 3-5 d | ⚑ **Hard gate.** If 2.45 ≠ 2.46 forward-compatible, either upgrade localai to 2.45 or wait for a Hub QAIRT 2.46 publish. |
| **1** | Compile SD 1.5 inpaint UNet via Hub; numerical parity check vs PyTorch reference (max-abs-diff < 1e-2 on float, ~1e-1 on w8a16 quant). | Working `sd15_inpaint_unet.bin` | 1-2 wk | ⚑ Quality gate (parity numbers). |
| **2** | Native pipeline: load VAE encoder, swap UNet load to inpaint variant, plumb 9-channel input, implement `runOutfitSwap` with a hand-crafted hardcoded mask (skip segmentation). End-to-end first generated outfit-swap PNG on-device. | First on-device result | 1-2 wk | |
| **3** | Wire SegFormer into the pipeline; implement mask_ops (classmap → binary → dilate → feather → downsample); garment-class selector logic with `garmentSpec` parsing. | Full auto pipeline on-device | 1 wk | |
| **4** | AIDL extension (`generateOutfitSwap`, `cancelOutfitSwap`, optional `onStage`); `LocalAiService` wiring; `OutfitSwapRunner`; LocalAiBridge `pickPhoto` + `generateOutfitSwap`. | Service-callable from RN | 3-5 d | |
| **5** | `GalleryPickerActivity`; `outfit-swap-1` RN widget with chip row, prompt input, progress, result rendering. | Shippable widget on dev device | 1 wk | |
| **6** | Quality tuning: mask dilation values, feather sigma, denoise strength (DDIM η, mask noise reblending strength), prompt prefix template ("portrait photograph of a person wearing {prompt}, same pose, photorealistic, …"), edge-blend post-processing if seams are visible. | Polished v1 | 2 wk | |

**Total: 7-10 weeks** to a shippable widget at enterprise pace.

---

## 11. Memory & latency budget

Per-call breakdown on Snapdragon 8s Gen 4 (estimates; refine empirically Phase 0):

| Stage | Model | Approx. params | Approx. binary | Time |
|---|---|---|---|---|
| Segment | SegFormer-B2-Clothes | 27.4M | ~30-60 MB w8a16 | 60-100 ms |
| Text encode | CLIP ViT-L/14 (existing) | 123M | ~125 MB | ~2 ms (existing path) |
| VAE encode | VAE encoder (existing asset) | ~34M | ~70 MB | ~50 ms |
| UNet diffuse | SD 1.5 inpaint UNet | 859M | ~1.5 GB w8a16 | ~50 ms × 20 steps = 1.0 s |
| VAE decode | VAE decoder (existing) | ~50M | ~100 MB | ~120 ms |
| **Total wall-clock (excluding I/O)** | | | | **~1.3-1.5 s** |

Peak HTP context residency at any moment: at most UNet (~1.5 GB) + mask_latent + masked_img_latent staging (<5 MB). Within Hexagon V79 budget. Other models swap in and out via context binary load/unload.

If unload latency is significant (~100-300 ms each), end-to-end wall-clock becomes ~2-3 s. Still acceptable; widget shows staged progress.

---

## 12. Risks & mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | QAIRT 2.45 binary fails to load under 2.46 runtime | Low | High (whole plan stalls) | Phase 0 settles this in 3-5 d. Fallback: pin localai to 2.45 or compile with QAIRT 2.46 if Hub catches up. |
| 2 | qai-hub Device enum lacks `Snapdragon 8s Gen 4` | Medium | Medium | Compile for `Snapdragon 8 Gen 3` and rely on forward compat (same Hexagon family within QAIRT major). |
| 3 | SD 1.5 inpaint UNet 9-channel first conv quantization regression | Low | Medium | Phase 1 numerical parity gate catches this. Fallback: leave the first conv in fp16 and only quantize the rest. |
| 4 | SegFormer transformer attention compiles but is slow on V79 | Low | Medium | Swap to U2Net cloth-segmentation (CNN-only, 4 classes). Clean drop-in. |
| 5 | mattmdjaga mask quality poor on real selfies (occlusions, hands across body) | Medium | Medium | Aggressive dilation + feather; in v1.1 add tap-to-paint mask refinement via SAM-2 (also on Hub catalog). |
| 6 | SD 1.5 base quality ceiling on complex garments (lace, brand logos) | High | Low/Medium | Accept for v1. v2 path: SDXL inpaint via Hub compile, or a small LoRA fine-tune on fashion images. |
| 7 | Free-text → garment-class mapping ambiguous ("an outfit") | Medium | Low | v1: fall back to "auto" selector. v1.1: route ambiguous prompts through Gemma 4 E4B for intent extraction. |
| 8 | Memory pressure with multiple context swaps causes OOM on lower-RAM SKUs | Low | Medium | Single-resident-at-a-time orchestration plus explicit `unload()` between stages. |
| 9 | Mask edges produce visible seams at the latent boundary | High | Low | Standard SD-inpaint trick: alpha-blend the final decoded image with the original outside the mask using the feathered mask. Implemented in Phase 6. |

---

## 13. Decision log

### 13.1 AIDL surface: focused vs generic
**Decision:** Focused `generateOutfitSwap` for v1. Sibling features (LaMa erase, SAM-driven object remove) get *new* AIDL methods later — append-only is the law.
**Revisit if:** We discover three or more image-edit operations all want the same parameters within one quarter.

### 13.2 Context-fill inpainter role (LaMa / AOT-GAN / MI-GAN)
**Decision:** No context-fill inpainter in v1 outfit-swap. The SD 1.5 inpaint UNet does its job in a single pass. If a future "Erase from photo" sibling widget is built, default to **MI-GAN** (picsart-ai-research/mi-gan) over LaMa — order-of-magnitude smaller and faster, MIT, and the published ONNX feeds directly into Hub compile.
**Revisit if:** Phase 6 quality tuning reveals that pre-erasing the garment before the inpaint UNet meaningfully improves prompt adherence (the inpaint UNet bleeds original garment colors/textures through into results). In that case, add MI-GAN as Stage 1 — not LaMa.

### 13.3 Base UNet vs inpaint UNet (you did not answer; defaulted)
**Decision:** Inpaint UNet only. It can degrade to plain text-to-image by passing an all-ones mask and a zero masked-image latent; existing `generateImage` callers route to the inpaint UNet that way. Halves the on-device UNet asset size.
**Revisit if:** Plain text-to-image quality degrades measurably with the inpaint UNet (rare in practice; the inpaint UNet is a fine-tune of base SD 1.5 and produces equivalent text-to-image samples).

### 13.4 Widget scope (you did not answer; defaulted)
**Decision:** Single widget (`outfit-swap-1`). No premature siblings. If we later ship `erase-1` or `remove-bg-1`, each gets its own AIDL method and its own RN bundle.
**Revisit if:** Product wants a unified "Photo edit" widget — easy retrofit since each operation is already isolated server-side.

### 13.5 Target chipset
**Decision:** Snapdragon 8s Gen 4, as specified.
**Revisit if:** Hub SDK doesn't list it as a device; fall back to 8 Gen 3 compile and verify on-device forward-compat.

---

## 14. Success criteria for v1

- A user can pick a portrait photo and a prompt ("a black hoodie") and within ~3 seconds see the same person in a black hoodie, with face, hair, and background unchanged on visual inspection.
- The same photo with different prompts produces different believable outfits.
- Top-of-funnel quality (random sample of 50 internal portraits): ≥60% rated "acceptable or better" by a small internal review, where "acceptable" means face untouched, mask edge not visually obvious at arm's length, and prompt fidelity directionally correct.
- End-to-end on-device, no cloud calls, no model downloads after install.
- Existing localai features (text gen, image gen, audio) continue to function unchanged.

---

## 15. Open questions (to settle during Phase 0)

1. Exact `qai_hub.Device` string for Snapdragon 8s Gen 4 — verify via `qai_hub.get_devices()` listing.
2. Does Hub's auto-quantization handle SegFormer's mix-of-FFN + attention cleanly w8a16, or do we need to mark certain layers fp16?
3. VAE encoder context binary: is the shipped asset Hub-published or a one-off conversion in the xororz bundle? Determines whether we re-compile it or just wire the existing loader.
4. Smallest input size SD 1.5 inpaint can run with acceptable quality. v1 plan assumes 512×512; if 384×384 cuts UNet step time by ~40% without obvious quality loss, ship that as default.

---

## Appendix A — File touch list

### localai (write/edit)
- `app/src/main/cpp/diffusion.cpp` — new loaders, `runOutfitSwap`, 9-channel UNet input plumbing
- `app/src/main/cpp/diffusion.hpp` — function signatures
- `app/src/main/cpp/mask_ops.cpp` *(new)* — mask pre/post processing
- `app/src/main/cpp/mask_ops.hpp` *(new)*
- `app/src/main/cpp/imagegen.cpp` — new JNI method `nativeRunOutfitSwap`
- `app/src/main/cpp/CMakeLists.txt` — add `mask_ops.cpp`
- `app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl` — append `generateOutfitSwap`, `cancelOutfitSwap`
- `app/src/main/aidl/com/nothing/localai/IImageGenCallback.aidl` — append `onStage`
- `app/src/main/java/com/nothing/localai/LocalAiService.kt` — `generateOutfitSwap` method
- `app/src/main/java/com/nothing/localai/OutfitSwapRunner.kt` *(new)* — runner lifecycle
- `app/src/main/java/com/nothing/localai/ImageGenNative.kt` — new external fn declaration
- `app/src/main/assets/segformer_b2_clothes.bin` *(new asset)*
- `app/src/main/assets/sd15_inpaint_unet.bin` *(new asset, replaces base UNet asset)*
- `tools/aihub-compile/` *(new directory)* — Python compile scripts
- `tools/aihub-compile/pyproject.toml`
- `tools/aihub-compile/compile_segformer_clothes.py`
- `tools/aihub-compile/compile_sd15_inpaint_unet.py`
- `tools/aihub-compile/validate_*.py`

### Aiwidget (write/edit)
- `android/app/src/main/java/com/nothing/aiwidget/localai/capture/GalleryPickerActivity.kt` *(new)*
- `android/app/src/main/java/com/nothing/aiwidget/localai/LocalAiBridge.kt` — `pickPhoto`, `generateOutfitSwap`
- `android/app/src/main/java/com/nothing/aiwidget/localai/LocalAiConnection.kt` — if any new AIDL plumbing requires it
- `android/app/src/main/AndroidManifest.xml` — declare `GalleryPickerActivity` not-exported
- `android/app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl` — mirror localai's append (the two repos keep the AIDL in sync)
- `android/app/src/main/aidl/com/nothing/localai/IImageGenCallback.aidl` — mirror append
- `testwidgets/outfit-swap-1/widget.json` *(new)*
- `testwidgets/outfit-swap-1/src/index.tsx` *(new)*
- `testwidgets/outfit-swap-1/src/styles.ts` *(new)*

---

## Appendix B — Garment selector → SegFormer class mapping

mattmdjaga class IDs (see model card):

| Selector | Class IDs (SegFormer) | Class names |
|---|---|---|
| **Top** | {4} | Upper-clothes |
| **Bottom** | {5, 6} | Skirt, Pants |
| **Dress** | {7} | Dress |
| **Full body** | {4, 5, 6, 7} | All garments above |
| **Auto** | {1, 4, 5, 6, 7, 8, 17} | Hat, Upper-clothes, Skirt, Pants, Dress, Belt, Scarf (deliberately excludes Face, Hair, Limbs, Bag, Shoes) |

Shoes (9, 10) intentionally excluded from v1 selectors — masking shoes alone tends to produce poor results without a dedicated shoe-shape prior. Hat (1) included in Auto only.
