# PLAN — img2img for Outfit Swap (Phase 6)

**Status (2026-05-16): SHIPPED + extended.** See "Shipped state" section at the bottom. The plan below is preserved as the original design doc.
**Goal:** Replace the txt2img + composite step in `OutfitSwapRunner` with true SD 1.5 img2img seeded from the MI-GAN-erased image, so generated garment texture warps to body shape and inherits perspective + lighting.

---

## 1. Where we are now

`PLAN-OUTFIT-SWAP.md` is the master plan. We shipped a working v1 hybrid pipeline (see [`README-OUTFIT-SWAP.md`](./README-OUTFIT-SWAP.md)):

```
input photo
   │
   ▼
SegFormer (LiteRT)  → garment classmap → 512×512 binary mask
   │
   ▼
MI-GAN (LiteRT)     → erased image: garment region context-filled with skin/bg
   │
   ▼
SD base UNet via existing nativeRunDiffusionPng (QNN) → txt2img from scratch
   │                                                    on prompt "close-up garment texture..."
   ▼
luminance-preserving composite (Kotlin) → final PNG
```

This works end-to-end but has a fundamental quality limit: txt2img generates a 512×512 image **from Gaussian noise**, so the generated texture has zero awareness of the original body's shape, perspective, or fold geometry. The composite then stamps that flat texture into the mask region. Even with luminance transfer, the texture grid scale is uniform across the body — it never folds with the shoulders or scales with depth.

**The fix:** img2img — start diffusion from the *MI-GAN-erased image's latent* with partial noise added. The UNet then transforms the existing body+context into the prompted outfit while respecting anatomy and perspective.

---

## 2. Architecture (new path)

Replace step 3 above (`nativeRunDiffusionPng`) with `nativeRunDiffusionImg2ImgPng`:

```
mi-gan-erased Bitmap (512×512 RGB)
   │
   ▼
[Kotlin]  decode → fp32 CHW [0,1] (3×512×512)
   │
   ▼
[native]  VAE encoder (xororz vae_encoder.bin, QNN) → z_0 (1×4×64×64 fp32)
   │                                                  scale by 0.18215
   ▼
[native]  noised init at timestep t_start = strength · T_max:
            z_t = √(α̅_t) · z_0 + √(1 − α̅_t) · ε     (fresh Gaussian ε, given seed)
   │
   ▼
[native]  scheduler.setTimesteps(iters) — full schedule
[native]  find inferenceStepIndex k where timestep(k) ≈ t_start
[native]  for i = k .. iters−1:   eps = base_unet(z, t_i, txt_emb_cfg); z = scheduler.step(eps, i, z)
   │
   ▼
[native]  vaeDecodeToPng (existing) → PNG bytes
   │
   ▼
[Kotlin]  OPTIONAL: hard-blend with original outside the SegFormer mask so face/hair are bit-exact
[Kotlin]  → return PFD
```

Strength controls how much of the original body survives:
- `strength = 1.0` → equivalent to current txt2img (pure noise init)
- `strength ≈ 0.7` → expected sweet spot: garment texture follows body, face/hair drift slightly
- `strength ≈ 0.4` → very faithful to original, garment changes subtly

Default ship with `strength = 0.7` and expose as a runner parameter for future tuning.

---

## 3. Concrete file changes

### 3.1 `app/src/main/cpp/diffusion.cpp` + `diffusion.hpp`

Add a new top-level function `runDiffusionImg2ImgToPng` that mirrors `runDiffusionToPng` but takes an input image and strength.

Signature (declare in `diffusion.hpp`):

```cpp
// Phase 6 outfit-swap img2img entry. Same SD 1.5 pipeline as runDiffusionToPng,
// but the latent is initialized by VAE-encoding `inputRgbFp32` (3*512*512 CHW,
// values in [0,1]) and adding noise at level √(1−α̅_t) where
// t = clamp(0.0, 1.0, strength) · 999.
//
// The diffusion loop runs only the tail of the schedule starting from the
// inference step whose timestep is closest to t — so iters=20 with strength=0.7
// actually runs ~14 UNet calls. Both eps cond + uncond and DDIM CFG identical
// to runDiffusion.
//
// On success `pngOut` holds the final composite PNG and `report` is the
// multi-line trace (same format as runDiffusion's report). Empty pngOut +
// non-empty report indicates failure.
bool runDiffusionImg2ImgToPng(const std::string&         bundleDir,
                              const std::vector<float>&  inputRgbFp32,
                              const std::string&         prompt,
                              float                      strength,
                              int                        iters,
                              uint64_t                   seed,
                              std::vector<uint8_t>&      pngOut,
                              std::string&               report);
```

Implementation reuses what already exists:

1. **Bundle / text encode / unet load** — identical to `runDiffusion`. Copy lines 1–250 (approx) of `diffusion.cpp`'s `runDiffusion` body.

2. **VAE-encode the input image** — the asset is `bundle.vaeEncoderBin` (already in `loadBundle()`'s output). The encoding logic is already written in `app/src/main/cpp/outfit_swap.cpp` as `vaeEncodeImage(vaeEncoderBin, rawRgb01CHW, latentChwOut, report)` (lines ~226–296). **Lift that function into diffusion.cpp** (or extract into a shared `vae_io.hpp/.cpp` if cleaner). Already handles:
   - VAE-input normalization `[0,1] → [-1,1]`
   - QnnSession init / inspect / instantiate on the encoder
   - NCHW/NHWC layout detection on input + output slots
   - Multiplies output by `kLatentScale = 0.18215f` (SD 1.5 convention)
   - Returns `latentChwOut` of length 16384 (1*4*64*64) CHW row-major

3. **Compute starting timestep and noise the latent**:

   ```cpp
   Scheduler sched;
   sched.setTimesteps(iters);
   const auto& abar = sched.alphasCumprod();   // length 1000

   const float clampedStrength = std::clamp(strength, 0.0f, 1.0f);
   const int   tStart = static_cast<int>(clampedStrength * 999.0f);

   // Find the inference step index whose timestep matches tStart.
   int kStart = 0;
   for (int i = 0; i < iters; ++i) {
       if (sched.timestep(i) <= tStart) { kStart = i; break; }
   }

   const float a = abar[sched.timestep(kStart)];
   const float sqrtA  = std::sqrt(a);
   const float sqrt1A = std::sqrt(1.0f - a);

   std::vector<float> eps(kLatentElems);
   fillGaussian(eps, seed);

   std::vector<float> latents(kLatentElems);
   for (int j = 0; j < kLatentElems; ++j) {
       latents[j] = sqrtA * z0[j] + sqrt1A * eps[j];
   }
   ```

   The scheduler's order-2 multistep history is reset by `setTimesteps`. The first call to `sched.step()` after a setTimesteps falls back to first-order (Euler) automatically — fine for img2img where the first step is just one of many in the tail.

4. **Run the diffusion loop, but starting at i = kStart**:

   ```cpp
   for (int i = kStart; i < iters; ++i) {
       const int t = sched.timestep(i);
       if (!runUnet(latents, t, embCondBytes,   noiseCond,   err))  return ...;
       if (!runUnet(latents, t, embUncondBytes, noiseUncond, err))  return ...;
       for (int k = 0; k < kLatentElems; ++k) {
           pred[k] = noiseUncond[k] + kGuidanceScale * (noiseCond[k] - noiseUncond[k]);
       }
       latents = sched.step(pred, i, latents);
       // … finite-check + logI same as runDiffusion …
   }
   ```

   `runUnet` is the existing helper inside `runDiffusion` — extract it into a file-scope helper so both txt2img and img2img can share it.

5. **VAE decode + PNG encode** — reuse `vaeDecodeToPng(bundleDir, latents, pngOut, vaeReport)` verbatim. No changes needed.

### 3.2 `app/src/main/cpp/imagegen.cpp` (JNI)

Add a new export mirroring `Java_..._nativeRunDiffusionPng` but with two extra args (input float array + strength):

```cpp
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeRunDiffusionImg2ImgPng(
        JNIEnv*      env,
        jobject      /*thiz*/,
        jstring      bundleDir,
        jstring      prompt,
        jfloatArray  inputRgbFp32,    // 3*512*512, CHW, [0,1]
        jfloat       strength,        // 0.0..1.0
        jint         iters,
        jlong        seed)
{
    // Pull jstring -> std::string (jstringToStdString helper exists)
    // GetFloatArrayRegion(inputRgbFp32) into std::vector<float>
    // Call imagegen::runDiffusionImg2ImgToPng(...)
    // Log report lines, return PNG bytes (mirror nativeRunDiffusionPng).
}
```

### 3.3 `app/src/main/java/com/nothing/localai/imagegen/NativeImageGen.kt`

Add the `external fun`:

```kotlin
/**
 * Phase 6 img2img entry — see PLAN-IMG2IMG.md. Given the MI-GAN-erased
 * RGB image as [inputRgbFp32] (3*512*512 CHW [0,1]) and a [strength] in
 * [0,1], runs SD 1.5 with the latent seeded from the input image at
 * `t = strength·999` instead of from pure noise. Returns PNG bytes or null.
 */
external fun nativeRunDiffusionImg2ImgPng(
    bundleDir: String,
    prompt: String,
    inputRgbFp32: FloatArray,
    strength: Float,
    iters: Int,
    seed: Long,
): ByteArray?
```

### 3.4 `app/src/main/java/com/nothing/localai/imagegen/OutfitSwapRunner.kt`

Replace the txt2img section with the img2img call. Key changes:

- After `MiganRunner.erase(...)` produces `erased512: Bitmap`, convert it to a flat fp32 CHW `[0,1]` buffer (size `3*512*512`).
- Call `NativeImageGen.nativeRunDiffusionImg2ImgPng(bundleDir, buildPrompt(prompt), erasedRgbFp32, /*strength=*/0.7f, iterations, seed)`.
- Decode the returned PNG → `gen512`.
- **Compositing stage becomes optional/lighter.** With img2img the entire result is already body-aware. You can either:
  - Use img2img output directly as the final result (simplest), OR
  - Still composite with original to preserve face/hair bit-exactly outside the SegFormer mask. **Recommend keeping the mask-blend** — img2img can subtly drift face features, and the SegFormer mask is precise enough that hard-stamping the original outside the mask is safe.

- `buildPrompt` should change back from texture-focused to outfit-focused. With img2img the body shape is preserved, so we WANT a "person wearing X" prompt — the network knows where the body is and will paint the garment onto it. Suggested:

  ```kotlin
  private fun buildPrompt(userPrompt: String): String {
      val cleaned = userPrompt.trim().trimEnd('.')
      if (cleaned.isEmpty()) return "portrait photograph, photorealistic"
      return "portrait of a person wearing $cleaned, photorealistic, " +
          "sharp focus, studio lighting, same pose, same face"
  }
  ```

### 3.5 `OutfitSwapProbe.kt`

No change needed — it calls into `OutfitSwapRunner.generate`, which will pick up the new path automatically.

---

## 4. Build & test loop

After the code changes:

```bash
cd ~/Documents/GitHub/localai
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell 'am force-stop com.nothing.localai.debug'
adb logcat -c
adb shell 'am start -n com.nothing.localai.debug/com.nothing.localai.ui.StatusActivity'

# Wait ~40-50s, then pull the result:
adb shell 'run-as com.nothing.localai.debug cat files/segformer-probe/outfit-swap-debug.png' > /tmp/img2img-result.png
```

The probe uses the same test JPEG already on device. Vary the prompt by editing `OutfitSwapProbe.run()`'s defaults (line ~16). Default test prompt is `"red plaid flannel shirt"`; once img2img works, this should produce a garment that *folds with the body* rather than a flat plaid sheet.

**Tuning knobs** to try if first result is off:
- `strength = 0.6` (more original preserved, less prompt adherence)
- `strength = 0.8` (more prompt adherence, body shape may drift)
- iters bump from 8 → 20 (sharper, ~50s instead of 25s)
- Re-introduce the SegFormer-mask hard-composite outside the mask to lock face/hair (already in OutfitSwapRunner as `compositeBitmaps` — pass `alpha=feathered, gen=img2imgResult, erased=originalPhoto`).

---

## 5. What's already on-device / on-disk

No new model compiles needed. All required artifacts exist:

| Artifact | Location | Status |
|---|---|---|
| `vae_encoder.bin` (QNN, xororz) | `filesDir/models/sd-v15-xororz/` | ✅ on device, never been loaded yet |
| `vae_decoder.bin` (QNN, xororz) | `filesDir/models/sd-v15-xororz/` | ✅ loaded by `vaeDecodeToPng` |
| `unet.bin` (QNN base SD UNet, xororz) | `filesDir/models/sd-v15-xororz/` | ✅ loaded by `runDiffusion` |
| `clip_v2.mnn` (text encoder) | `filesDir/models/sd-v15-xororz/` | ✅ loaded by MNN session |
| `segformer_b2_clothes.tflite` (LiteRT) | `filesDir/segformer-probe/` | ✅ pushed via run-as |
| `migan.tflite` (LiteRT) | `filesDir/segformer-probe/` | ✅ pushed via run-as |
| `test_portrait.jpg` (probe input) | `filesDir/segformer-probe/` | ✅ pushed via run-as |

`outfit_swap.cpp` already has `vaeEncodeImage()` (~70 lines) that you should reuse / lift directly. The function is currently dead code — it was the SD 1.5 inpaint UNet path that we abandoned when Hub couldn't compile the inpaint UNet.

---

## 6. Gotchas learned this session

1. **Hub QNN compile produces blobs this device's runtime won't load** (`rc=0x138d`). Anything QNN-target needs to come from the xororz toolchain. The vae_encoder.bin in the bundle IS from xororz — it'll load fine. Don't try to recompile it.
2. **Scheduler's `step()` after `setTimesteps()` defaults to first-order on the first call** (no `x0History_`). For img2img, this is exactly what we want for step `kStart` — the previous steps "didn't happen" from the solver's perspective.
3. **VAE-encoder I/O layouts vary.** xororz's vae_decoder is NCHW; vae_encoder may differ. `vaeEncodeImage` already handles NCHW vs NHWC via `classify4D`. Trust the existing detection.
4. **Native build complains aren't deal-breakers.** clangd in this repo doesn't know about NDK headers (`android/log.h`, `jni.h`) and reports false errors. The actual NDK build via `./gradlew :app:assembleDebug` will surface real issues. Ignore clangd noise.
5. **Boot probe runs everything sequentially.** `LocalAiApp.probeQnnInspectIfBinaryPresent()` runs UNet probe → MNN probe → SD txt2img probe (8 iters, ~17s) → SegformerProbe → MiganProbe → OutfitSwapProbe. Expect ~50s from app launch before the outfit-swap PNG is on disk.
6. **APK ships QAIRT 2.46 libs** (`libQnnHtpV73Stub.so` etc., bundled via `stageQnnLibs` task in `app/build.gradle`). The device's QNN core API version reported as `v2.35.0` is the runtime's *interface* version, not the SDK version — don't be confused by the mismatch.
7. **Storage is tight on dev machine** (~5 GiB free as of session end). Don't recompile anything to AI Hub. If you do need to, clean `app/src/main/assets/*.tmp.*` first.

---

## 7. Acceptance criteria

img2img is "good enough" to ship as v1.1 when, on the probe's standard test JPEG with prompt `"red plaid flannel shirt"`:

- ✅ The plaid pattern visibly **folds with the body** at the shoulder line and collar (not uniform grid stamp).
- ✅ The shading inside the garment region follows the original lighting (highlights on top of shoulder, shadow under collar).
- ✅ Face/hair outside the SegFormer mask are unchanged (bit-exact via the optional hard-composite, or visually indistinguishable if drift is small).
- ✅ Pipeline total time ≤ 40s at iters=12, strength=0.7.

If the result is still flat-looking at strength=0.7, drop to strength=0.55 and try `iters=20`.

---

## 8. Out of scope (defer to later)

- Re-quantizing the base UNet for speed.
- Multi-person photo handling (SegFormer masks all subjects' garments at once).
- True inpaint UNet (the original §13.3 plan path — Hub couldn't compile it; revisit when a local Linux toolchain is available).
- Widget UI changes — current widget already chains through `OutfitSwapRunner.generate`, so the img2img improvement requires no widget rebuild.

---

## 9. Estimated effort

- 60-90 min of native code edits (mostly mechanical refactor + extract `vaeEncodeImage` + write img2img orchestrator).
- 10-15 min Kotlin runner wiring.
- 1-3 build / install / probe cycles to tune strength and verify.

Total: about 2 hours of focused work for a working v1.1.

---

## 10. Shipped state (2026-05-16)

v1.1 went in as designed, with two real-world deltas worth documenting:

### Deltas vs the original plan

1. **xororz `vae_encoder.bin` has 2 outputs, not 1.** It exports the posterior's `mean` and `logvar` separately (`AutoencoderKL.encode().latent_dist`). `vaeEncodeImage` in `diffusion.cpp` was relaxed to pick the first 4-channel 64×64 output (the mean); at strength≥0.7 the added Gaussian noise dominates the encoder's per-pixel stochasticity, so skipping `logvar` is fine.
2. **OutfitSwapRunner composite logic.** The original luminance-preserving composite (`compositeBitmaps`) was designed for the txt2img path where the generated image is a flat fabric texture. img2img output already carries body shading, so a new `compositeAlphaSimple` (straight feathered alpha blend, no luminance transfer) replaced it for the img2img path. `compositeBitmaps` is kept (marked `@Suppress("unused")`) as historical reference.

### Extension shipped on top: RePaint latent blending

Base img2img at strength=0.7 still produced two visible failure modes on real portraits:
- **Anatomy drift outside the mask** — SD's prior would happily redraw the hand-near-face into "extra hand" or change leg geometry.
- **Boundary seams** — the latent on either side of the SegFormer mask diverged through diffusion, so the VAE-decoded image had visible color/tone discontinuities at the boundary even after the Kotlin alpha-composite.

Both are addressed by RePaint-style latent blending applied inside the diffusion loop:

```cpp
// After scheduler.step(pred, i, latents):
const int   tBlend  = (i + 1 < iters) ? sched.timestep(i + 1) : 0;
const float ab      = abar[tBlend];
const float sA      = std::sqrt(ab);
const float s1A     = std::sqrt(std::max(0.0f, 1.0f - ab));
fillGaussian(stepNoise, blendSeedBase + uint64_t(i) * 0x9E3779B97F4A7C15ULL);
for (int c = 0; c < kLatentChan; ++c) {
    const std::size_t base = c * kLatentH * kLatentW;
    for (std::size_t j = 0; j < kLatentH * kLatentW; ++j) {
        const float m     = mask64Fp32[j];
        const float zOrig = sA * z0[base + j] + s1A * stepNoise[base + j];
        latents[base + j] = m * latents[base + j] + (1.0f - m) * zOrig;
    }
}
```

The 64×64 mask is broadcast across all four latent channels. Outside-mask latents at every step are bit-exact a noised copy of `z₀` at the next timestep's noise level → no drift, no seam. Inside-mask latents carry the full diffused signal → free to regenerate the garment.

**JNI surface change.** `nativeRunDiffusionImg2ImgPng` gained a `mask64Fp32: FloatArray` parameter between `inputRgbFp32` and `strength`. The Kotlin side (`OutfitSwapRunner`) feathers the 512×512 binary mask at σ=2 (sharper than the σ=4 used for the final composite), then mean-pools 8× down to 64×64 via `downsampleMaskMean(...)`.

Pass an all-ones mask to disable blending (pure img2img). Pass an all-zeros mask to round-trip the input image (z₀ VAE-decoded).

### Verified metrics

- `mask64 hits=885/4096` for the test portrait (22% regenerate, 78% locked).
- img2img loop time unchanged at ~6.4s @ iters=12 — blending overhead is ~10µs per step, dwarfed by UNet calls.
- Total img2img stage: ~14.9s.
- End-to-end probe time (`OutfitSwapProbe.run`): ~26s.

### Known residual issue

Even with latent blending, a faint seam may remain at the mask boundary in image space. Cause is **VAE round-trip blur** — outside-mask uses the sharp original photo (via Kotlin composite), inside-mask uses VAE-decoded pixels (softer). See `README-OUTFIT-SWAP.md` "Recommended next-step interventions" for the Poisson / color-match / IP-Adapter ladder.
