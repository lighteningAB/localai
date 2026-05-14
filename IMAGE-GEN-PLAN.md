# Image Generation Implementation Plan (Plan B — clean-room QNN)

This is a self-contained resume point. The wiring (AIDL → service → bridge → React Native widget) is done and on-device. The native inference engine is partially built: Phases 1, 2, 3, 5 are landed and verified; Phase 4 is split into 4a (QNN — blocked on QAIRT install) and 4b (MNN text encoder — unblocked); Phases 6–8 chain on Phase 4.

**Tell the new session:** "Read `IMAGE-GEN-PLAN.md`. Phases 1/2/3/5 are done — see the status table in §3. Don't re-walk the LiteRT or MediaPipe paths — `KNOWN-ISSUES.md` documents why both were abandoned. Plan B is clean-room native C++ + QNN/MNN, no Local Dream code is copied."

---

## 1. State of the world

### Works end-to-end
- AIDL contract — `app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl` has `generateImage(prompt, iterations, seed, cb): String` and `cancelImageGen(rid)`. Mirrored byte-identical in `Aiwidget/android/app/src/main/aidl/com/nothing/localai/ILocalAiService.aidl`. AIDL is append-only — never reorder.
- AIDL callback — `IImageGenCallback.aidl` (mirrored both repos): `onStep(rid, step, total)`, `onResult(rid, ParcelFileDescriptor pngFd, w, h)`, `onError(rid, code, msg)`. All `oneway`.
- Service wiring — `LocalAiService.kt` (~140–180) wraps the callback, calls `beginRequest()/endRequest()` for the foreground-notification lifecycle.
- React Native bridge — `Aiwidget/.../LocalAiBridge.kt`: `generateImage(prompt, iters, seed, promise)` emits `LocalAi.image` events `{ requestId, step?, totalSteps?, uri?, width?, height?, error?, done }`. PFD copied to Aiwidget cacheDir; widget reads via `file://` URI.
- Widget — `Aiwidget/testwidgets/image-gen-1/src/App.tsx`. Landscape UI, prompt + iters spinner.

### Native engine, partially built
- `app/src/main/cpp/` exists. Verified on-device probe: `LocalAiApp.onCreate()` calls `NativeImageGen.nativePing()`, logcat shows `imagegen native loaded: ping=imagegen-native-v0`.
- Scheduler, bundle loader, CLIP tokenizer cross-compile into `libimagegen.so` and pass host-side tests against real data (xororz bundle, Python `tokenizers` reference).

### Stubbed (still TODO)
- `app/src/main/java/com/nothing/localai/imagegen/ImageGenRunner.kt` — still returns `BUNDLE_MISSING` or `NOT_IMPLEMENTED`. Phase 8 replaces the `NOT_IMPLEMENTED` branch with `NativeImageGen.nativeGenerate(...)`.
- No `qnn_session.{hpp,cpp}` yet (Phase 4a, blocked on QAIRT).
- No `mnn_session.{hpp,cpp}` yet (Phase 4b, unblocked but not started).
- No `imagegen.cpp` diffusion-loop logic beyond the ping stub.

---

## 2. Architecture (Plan B — what we actually build)

**Stack** (revised after Phase 2 bundle inspection):
- **UNet + VAE runtime**: Qualcomm AI Engine Direct SDK (QAIRT, formerly QNN SDK), version 2.39+ — proprietary but free dev license. Provides `libQnn*.so` for Hexagon V68 through V81. xororz packs UNet + VAE as QNN context binaries (`unet.bin`, `vae_encoder.bin`, `vae_decoder.bin`).
- **Text encoder runtime**: Alibaba MNN (Apache 2.0) — **primary path, not just fallback**. xororz ships CLIP as `clip_v2.mnn`, not as a QNN binary. MNN provides CPU + OpenCL backends; we use OpenCL on Adreno.
- **Tokenizer**: hand-rolled C++ CLIP byte-level BPE in `tokenizer.cpp`. Parses xororz's `tokenizer.json` (vocab + merges) via vendored `nlohmann/json` (single header, MIT). No Rust dep. Bit-exact against Python `tokenizers` for the test prompts in `tokenizer_test.cpp`.
- **Image encode**: stb_image_write.h (Public Domain) — for PNG output. (Not yet vendored; Phase 7.)
- **Compression**: not needed in-process — xororz bundles are plain zip; the user extracts on the host and pushes the directory.

**Model source**: xororz HF (https://huggingface.co/xororz/sd-qnn) — pre-converted SD 1.5 models packaged for QNN. Each model has 3 variants (filename pattern `<Name>_qnn2.28_<variant>.zip`):
- `_8gen1.zip` (~1 GB) — Hexagon V69 target
- `_8gen2.zip` (~1 GB) — **Hexagon V73 target → this is our Snapdragon 8s Gen 4**
- `_min.zip` (~1 GB) — fallback for older silicon

Inner layout (verified for `AbsoluteReality_qnn2.28_8gen2.zip`, see Phase 2):
```
output_512/qnn_models_8gen2/
  tokenizer.json         (~3.5 MB)   HuggingFace CLIP BPE
  clip_v2.mnn            (~149 MB)   text encoder, MNN format
  pos_emb.bin / token_emb.bin        CLIP embedding tables
  vae_encoder.bin / vae_decoder.bin  QNN context binaries
  unet.bin               (~840 MB)   QNN context binary, the heavy file
  *.patch                            resolution-specific UNet overlays (768, 1024, 768x1024, …)
```

These ship under their original SD CreativeML licenses (downstream of CompVis SD 1.5), separate from Local Dream's CC BY-NC application license. Personal/hobby use is unambiguously fine.

**License boundary (important):**
- Reuse: upstream libs (QNN SDK, MNN, nlohmann/json, stb), xororz HF model files, public reference implementations of schedulers/CFG/etc.
- Do NOT reuse: any code from `xororz/local-dream`. Its CC BY-NC covers source code, headers, and patches as a unit. Read for understanding what upstream APIs to call (not copyrightable), do not port their organization, naming, or structure.

**Project layout (as built):**
```
app/
├── src/main/cpp/
│   ├── CMakeLists.txt              # tolerant of missing QNN_SDK_ROOT
│   ├── imagegen.cpp                # JNI entry points (currently: nativePing only)
│   ├── scheduler.{hpp,cpp}         # DPM-Solver multistep (Phase 5 ✅)
│   ├── bundle_loader.{hpp,cpp}     # manifest validator (Phase 2 ✅)
│   ├── tokenizer.{hpp,cpp}         # hand-rolled CLIP BPE (Phase 3 ✅)
│   ├── qnn_session.{hpp,cpp}       # PHASE 4a — TODO
│   ├── mnn_session.{hpp,cpp}       # PHASE 4b — TODO
│   ├── 3rdparty/
│   │   └── nlohmann/json.hpp       # JSON, MIT, vendored single header
│   └── tests/
│       ├── run_host_tests.sh
│       ├── scheduler_test.cpp
│       ├── bundle_loader_test.cpp
│       └── tokenizer_test.cpp
├── build/qnn-libs/arm64-v8a/       # populated by stageQnnLibs Gradle task
│   ├── libQnnHtp.so                # from ${QNN_SDK_ROOT}/lib/aarch64-android/
│   ├── libQnnHtpV73Stub.so + libQnnHtpV73Skel.so   # our chip's Hexagon revision
│   └── …                           # all other V68..V81 stubs/skels (run-time selected)
└── src/main/java/com/nothing/localai/imagegen/
    ├── ImageGenRunner.kt           # exists — Phase 8 swaps NOT_IMPLEMENTED for nativeGenerate
    └── NativeImageGen.kt           # exists — only nativePing for now
```

Note: QNN libs go into `build/qnn-libs/` (via `stageQnnLibs` Gradle Copy task + an additional `jniLibs.srcDirs` entry), NOT `src/main/jniLibs/`. Keeps the ~50 MB proprietary binaries out of the source tree.

---

## 3. Phased implementation

| Phase | Title                                  | Status                  | Verified by                                                  |
|-------|----------------------------------------|-------------------------|--------------------------------------------------------------|
| 1     | Native toolchain skeleton              | ✅ done                 | on-device logcat: `imagegen native loaded: ping=imagegen-native-v0` |
| 2     | Model bundle loader                    | ✅ done                 | `bundle_loader_test.cpp` against the real 1 GB xororz zip    |
| 3     | CLIP BPE tokenizer                     | ✅ done                 | `tokenizer_test.cpp`: 5/5 prompts match Python `tokenizers`  |
| 4a    | QNN runtime (UNet + VAE)               | ✅ done                 | on-device: graph metadata + first synthetic forward run      |
| 4b    | MNN text encoder                       | ✅ done                 | on-device: `clip_v2.mnn` forward, first8 logged              |
| 5     | DPM-Solver multistep scheduler         | ✅ done                 | `scheduler_test.cpp`: alphas + zero-noise stability          |
| 6     | UNet loop + classifier-free guidance   | ✅ done                 | on-device: 5-step diffusion of "a cat" → finite latents in 4.1s loop, total 12.1s incl. text encode + qnn init |
| 7     | VAE decode + PNG encode                | ⛓ blocked on 6         | —                                                            |
| 8     | JNI bridge + ImageGenRunner wire-up    | ⛓ blocked on 7         | —                                                            |

### 3.1 Native toolchain skeleton (Phase 1) — ✅ done

**Deliverable:** `libimagegen.so` builds, links, dlopens on-device, and `NativeImageGen.nativePing()` round-trips a hardcoded string.

What was built:
- `app/src/main/cpp/CMakeLists.txt` (tolerant of empty/missing `QNN_SDK_ROOT`; emits a warning, still produces a working `libimagegen.so`).
- `app/src/main/cpp/imagegen.cpp` — single JNI symbol `Java_com_nothing_localai_imagegen_NativeImageGen_nativePing`.
- `app/src/main/java/com/nothing/localai/imagegen/NativeImageGen.kt` — `System.loadLibrary("imagegen")` + `external fun nativePing()`.
- `app/build.gradle` — `externalNativeBuild` wired; `qnn.sdk.root` resolved from env `QNN_SDK_ROOT` or `local.properties`; `stageQnnLibs` Gradle Copy task stages QNN libs into `build/qnn-libs/arm64-v8a/`; `jniLibs.srcDirs += build/qnn-libs` so AGP packages them.
- `LocalAiApp.onCreate()` calls `nativePing()` on app start and logs the result.

### 3.2 Model bundle loader (Phase 2) — ✅ done

**Deliverable:** validate a directory of extracted xororz bundle files; return absolute paths in a `Bundle` struct.

Design decision: bundles are plain zip. We pre-extract on the host and push the directory; no in-process zip extraction. The Phase 2 deliverable in the original plan ("decompress to memory or temp dir") was overkill — a manifest validator is sufficient.

Key files:
- `app/src/main/cpp/bundle_loader.{hpp,cpp}` — `loadBundle(rootDir, &out, &error)` returns true if all required files exist.
- `app/src/main/java/com/nothing/localai/imagegen/ImageGenRunner.kt` — `DEFAULT_DIFFUSION_DIR_NAME = "sd-v15-xororz"`; `isReady()` is a cheap pre-flight matching the native loader's required-files list.
- `scripts/push-diffusion-bundle.sh` — default `DEST_NAME = "sd-v15-xororz"` (kept in sync with the Kotlin constant).

Workflow:
```bash
curl -L -o /tmp/AbsoluteReality_qnn2.28_8gen2.zip \
    https://huggingface.co/xororz/sd-qnn/resolve/main/AbsoluteReality_qnn2.28_8gen2.zip
unzip /tmp/AbsoluteReality_qnn2.28_8gen2.zip -d /tmp/bundle
./scripts/push-diffusion-bundle.sh /tmp/bundle/output_512/qnn_models_8gen2
```

### 3.3 CLIP BPE tokenizer (Phase 3) — ✅ done

**Deliverable:** `Tokenizer::encode("a cat") → IntArray(77)` matching the Python `tokenizers` library output for the bundled `tokenizer.json`.

Design decision: hand-rolled, no Rust dependency. The original plan called for `mlc-ai/tokenizers-cpp` (wraps HF Rust tokenizers); rejected because it would force a permanent `rustup + aarch64-linux-android target` build dep. The hand-rolled implementation:
- Loads `tokenizer.json` via vendored `nlohmann/json` (single header, 25K lines, MIT).
- Implements byte-level BPE end-to-end: normalize (ASCII fast-path for whitespace + lowercase), pre-tokenize via the CLIP regex (ASCII `\p{L}/\p{N}` + multibyte UTF-8 treated as letter), `bytes_to_unicode` GPT-2-style encoder, greedy BPE merges with `</w>` suffix, vocab lookup, BOS/EOS wrap, pad/truncate to 77.
- Limitations: NFC normalization and Unicode-aware lowercase are not implemented; non-ASCII inputs may diverge slightly from the Python reference. Sufficient for English SD prompts (the common case).

Verification (`tokenizer_test.cpp`, against Python `tokenizers` output for the same `tokenizer.json`):
- `"a cat"` → `[49406, 320, 2368, 49407, ...EOS]`
- `"a photo of a cat"` → `[49406, 320, 1125, 539, 320, 2368, 49407, ...]`
- `""` → `[49406, 49407, ...EOS-padded]`
- `"A Cat!"` (case + punctuation) → `[49406, 320, 2368, 256, 49407, ...]`
- `"the brown fox jumps over the lazy dog"` → `[49406, 518, 2866, 3240, 18911, 962, 518, 10753, 1929, 49407, ...]`

### 3.4 Runtime initialization (Phase 4) — split into 4a / 4b

The original plan bundled QNN + text encoder into one phase. Reality is that xororz ships the text encoder as MNN, not QNN — they're separate runtimes loaded independently. Splitting clarifies dependencies and unblocks parallel work.

#### 3.4a QNN runtime for UNet + VAE — 🚫 blocked on QAIRT install

**Deliverable:** `QnnSession` class that owns device/context/graph/IO tensors; loads `unet.bin` (or `vae_decoder.bin`) as a context binary; runs one forward pass; returns output buffers.

Reference: `${QNN_SDK_ROOT}/examples/QNN/SampleApp/` for the API patterns (`QnnInterface_t`, `QnnContext_*`, `QnnGraph_*`, `QnnTensor_t`). **Don't copy Local Dream's wrappers** — write our own.

Hexagon revision detection on first init: `QnnHtpDevice_getInfrastructure()` returns arch info. On the Snapdragon 8s Gen 4 we expect V73. If wrong (V75/V77), fall back to the `_min.zip` variant.

Verification: load `unet.bin`, query input/output tensor metadata, dump shapes to logcat. Then run an inference with synthetic zero inputs and confirm the output is finite. Numerical correctness comes in Phase 6 when wired into the full loop.

#### 3.4b MNN text encoder — 🟡 unblocked

**Deliverable:** `MnnSession` class that loads `clip_v2.mnn`, runs forward on int32[1,77] → fp32[1,77,768], returns output buffer.

Add Alibaba MNN (Apache 2.0, https://github.com/alibaba/MNN) as a submodule under `app/src/main/cpp/3rdparty/MNN/`. Build into `libimagegen.so` via `add_subdirectory(3rdparty/MNN ...)`. MNN's CMake supports an Android cross-compile out of the box (`-DMNN_BUILD_FOR_ANDROID=ON -DMNN_OPENCL=ON`).

Pos/token embedding handling: `pos_emb.bin` and `token_emb.bin` are likely fp16 lookup tables consumed as part of CLIP's embedding layer. We may need to multiply / concatenate them with `clip_v2.mnn`'s output, or the `.mnn` model may already wrap them — verify by inspecting MNN model's input shape (`MNN::Interpreter::getSessionInputAll`).

Verification: tokenize `"a cat"` → run MNN forward → dump first 8 output values to logcat. Compare against a Python reference run using HuggingFace `transformers.CLIPTextModel`. Numerical drift within fp16 (~1e-3) is fine.

### 3.5 DPM-Solver multistep scheduler (Phase 5) — ✅ done

**Deliverable:** pure-C++ scheduler that takes `(eps, t_index, x_t)` and returns `x_{t-1}`.

Implemented in `scheduler.{hpp,cpp}` as DPM-Solver++ multistep (2nd-order, midpoint variant). Uses the natural alpha-cumprod parameterization (not karras-sigma form). 1st-order for the first inference step, 2nd-order multistep for subsequent steps.

Verification (`scheduler_test.cpp`):
- `alphas_cumprod[0]   = 0.99915` (expected ≈ 0.999)
- `alphas_cumprod[999] = 0.00466` (the well-known SD 1.5 value; the original plan said ≈0.06 which was off by an order of magnitude)
- `alphas_cumprod` monotonically decreasing
- Zero-noise step on zero-latent returns zeros (no NaN/Inf)
- 20-step run on synthetic deterministic-noise input stays finite
- Inference timesteps span 999..0 for `setTimesteps(20)`

### 3.6 UNet loop + CFG (Phase 6) — ⛓ blocked

**Deliverable:** running N UNet steps produces non-NaN latents.

The loop:
```cpp
auto latents = gaussianNoise({1, 4, 64, 64}, seed);
auto embCond   = mnn.run(tokenizer.encode(prompt));
auto embUncond = mnn.run(tokenizer.encode(""));
scheduler.setTimesteps(iters);
for (int i = 0; i < iters; ++i) {
    int t = scheduler.timestep(i);
    auto noiseCond   = qnnUnet.run(latents, t, embCond);
    auto noiseUncond = qnnUnet.run(latents, t, embUncond);
    auto pred = noiseUncond + GUIDANCE_SCALE * (noiseCond - noiseUncond);
    latents = scheduler.step(pred, i, latents);
    cb.onStep(rid, i + 1, iters);
}
```

Inspect UNet's input shape via QNN tensor metadata before assuming non-batched. If `(2,4,64,64)`, batch cond+uncond.

**Reuse the UNet session across iterations.** Construction is expensive; inference is fast.

**GPU/NPU contention**: Gemma 4 E4B holds an OpenCL context for LLM streaming, MNN text-encode will also use OpenCL, and UNet/VAE will hold an HTP context. They don't share a delegate but share device-level VRAM/RAM. Gate at the service layer (reject `generateImage` while a token stream is active) — already a TODO in `LocalAiService.kt`.

### 3.7 VAE decode + PNG (Phase 7) — ⛓ blocked

**Deliverable:** PFD containing a viewable PNG.

```cpp
latents *= (1.0f / 0.18215f);     // SD 1.5 latent scale factor
auto rgb = qnnVae.run(latents);    // float16[1,3,512,512]
// denorm: rgb = (clamp(rgb, -1, 1) + 1) * 127.5 → uint8
// channels-first → channels-last for stb_image_write
stbi_write_png_to_func(writeFn, &pfdCtx, 512, 512, 3, pixels, 512*3);
```

`writeFn` writes to the PFD write-end (passed in from JNI). The Kotlin side reads from the read-end (already wired in `ImageGenRunner.kt`/`LocalAiService.kt`).

### 3.8 JNI bridge (Phase 8) — ⛓ blocked

**Deliverable:** `ImageGenRunner.generate()` calls into native, callbacks land back in Kotlin.

JNI surface (to expand `NativeImageGen.kt`):
```kotlin
object NativeImageGen {
    init { System.loadLibrary("imagegen") }
    external fun nativePing(): String                     // already exists (Phase 1)
    external fun nativeInit(modelDir: String): Long       // returns engine handle
    external fun nativeGenerate(
        engine: Long, prompt: String, iters: Int, seed: Long,
        callback: NativeCallback,
    ): String                                              // returns rid
    external fun nativeCancel(engine: Long, rid: String)
    external fun nativeClose(engine: Long)
}
interface NativeCallback {
    fun onStep(rid: String, step: Int, total: Int)
    fun onResult(rid: String, pfdFd: Int, w: Int, h: Int)
    fun onError(rid: String, code: String, msg: String)
}
```

In `ImageGenRunner.generate()`, replace the `NOT_IMPLEMENTED` stub with `NativeImageGen.nativeGenerate(...)`, marshaling the AIDL callback through `NativeCallback`. PFD fd transfer: dup the fd in JNI, pass as int, reconstruct on the Java side as `ParcelFileDescriptor.fromFd()`.

---

## 4. Verification gates

- ✅ Phase 1 (toolchain): on-device `nativePing()` returns `imagegen-native-v0`.
- ✅ Phase 2 (bundle): `bundle_loader_test` against the real extracted xororz zip finds all 7 required files + at least one `.patch`.
- ✅ Phase 3 (tokenizer): `tokenizer_test` matches Python `tokenizers` output bit-for-bit on 5 reference prompts.
- ⏳ Phase 4a (QNN): load `unet.bin`, dump input/output tensor metadata to logcat; run synthetic forward → finite outputs.
- ⏳ Phase 4b (MNN): tokenize → MNN forward → first 8 output values match `transformers.CLIPTextModel` within fp16 drift (~1e-3).
- ✅ Phase 5 (scheduler): `alphas_cumprod[0/999]` correct; zero-noise step is the identity scaling; 20-step run on synthetic input stays finite.
- ⏳ Phase 6 (UNet): first iteration's noise_pred is finite; logcat shows step 1/N → N/N.
- ⏳ Phase 7 (VAE): `adb pull` the PFD output; `file` reports PNG; `open` shows an image (quality irrelevant).
- ⏳ End-to-end: tap generate in widget → image appears. Time it.

Host-side tests run via `./app/src/main/cpp/tests/run_host_tests.sh` — completes in <2 s when the bundle zip is present at `/tmp/AbsoluteReality_qnn2.28_8gen2.zip` (set `IMAGEGEN_TEST_BUNDLE_ZIP` to override).

---

## 5. Build / install / deploy reference

```bash
# Build + install localai. APK is com.nothing.localai.debug.
cd ~/Documents/GitHub/localai
./gradlew :app:installDebug

# Force clean if Gradle gets confused
./gradlew :app:clean :app:installDebug

# Run host-side C++ unit tests (scheduler, bundle_loader, tokenizer)
./app/src/main/cpp/tests/run_host_tests.sh

# Push an extracted xororz model bundle (default DEST_NAME = sd-v15-xororz)
unzip /tmp/AbsoluteReality_qnn2.28_8gen2.zip -d /tmp/bundle
./scripts/push-diffusion-bundle.sh /tmp/bundle/output_512/qnn_models_8gen2

# Push the testwidget bundle (rebuilds JS via Metro)
~/Documents/GitHub/essential-apps/scripts/device/push-testwidget.sh \
    ~/Documents/GitHub/Aiwidget/testwidgets/image-gen-1

# Tail logs (per-PID filter is broken on this device — use tag filter)
adb logcat -c
adb shell "logcat -v time -s imagegen:V ImageGenRunner:V LocalAiService:V LocalAiBridge:V LocalAiApp:V *:S"
```

QAIRT path configuration (Phase 4a onwards): set either env var or local property — whichever is set first wins.
```bash
# env var (per shell):
export QNN_SDK_ROOT=/absolute/path/to/qairt/2.39.x

# OR in local.properties (uncommitted, persistent):
echo "qnn.sdk.root=/absolute/path/to/qairt/2.39.x" >> local.properties
```

When configured, the `stageQnnLibs` Gradle task copies `libQnn*.so` (from `${QNN_SDK_ROOT}/lib/aarch64-android/`) and per-Hexagon Skel libs (from `${QNN_SDK_ROOT}/lib/hexagon-v{68,69,73,75,79,81}/unsigned/`) into `build/qnn-libs/arm64-v8a/`. AGP packages them via the additional `jniLibs.srcDirs` entry in `app/build.gradle`.

---

## 6. Gotchas already encountered (don't relearn these)

- **Gradle UP-TO-DATE lies**: `installDebug` returned exit 0 with a stale APK after a Kotlin compile error in `ImageGenRunner.kt`. Check `ls -lh` mtime on the APK after a build, or just `./gradlew :app:clean :app:installDebug` when suspect.
- **logcat per-PID is empty for our app processes** on this Nothing build. Use `logcat -v time -s <tag>:V` from the device shell, or filter by tag from the host. `adb logcat --pid=<PID>` silently returns nothing.
- **Foreground service notification**: `LocalAiService.beginRequest()` calls `startForegroundCompat()` with `FOREGROUND_SERVICE_TYPE_DATA_SYNC`. Image gen takes minutes — the notif must show or Android may kill the process.
- **GPU/HTP contention with LiteRT-LM**: Gemma 4 E4B holds OpenCL on `Backend.GPU()`. MNN text-encode will also use OpenCL. The SD UNet will hold Hexagon HTP. They don't share delegates but they share device-level memory budget. Gate at the service layer (reject `generateImage` while a token stream is active in `LocalAiService.kt`). Currently lenient.
- **Disk on dev Mac stays tight** (~10–20 GB free). xororz bundles are ~1 GB each — download incrementally and clean up `/tmp` after each phase. The host-test harness extracts the bundle into `${TMPDIR}/imagegen-host-tests/` on demand; rm it when done.
- **AIDL is append-only**: never reorder/remove methods in `ILocalAiService.aidl`. New methods at the end. Mirror byte-identical in Aiwidget.
- **AI Hub TFLite is impossible for SD at w8a16**: `python -m qai_hub_models.models.stable_diffusion_v*.export --target-runtime tflite` argparse-refuses for any SD model. Skipping AI Hub entirely is the right call — see `KNOWN-ISSUES.md`.
- **Original plan misnamed the xororz bundle URL**: pattern is `<Name>_qnn2.28_<variant>.zip` (note the `_qnn` prefix on the version), not `<Name>_v2.28_<variant>.zip`.
- **`alphas_cumprod[999]` for SD 1.5 scaled_linear is ~0.0047, NOT 0.06** (an earlier draft of this plan claimed 0.06).
- **CLIP text encoder in xororz bundles is MNN, not QNN**. The original plan framed MNN as a CPU fallback; it's actually part of the primary path for text encoding.
- **On SM8735 (Sun, "8s Gen 4", Hexagon V73) the HTP requires explicit opt-in to unsigned PD** for debug-signed (untrusted) apps. Without it, `QnnDevice_create` returns 0x36b1 (`QNN_DEVICE_ERROR_INVALID_CONFIG`) — but the actual root cause shows only in the kernel dmesg: `glink-edge.fastrpcglink-apps-dsp: Error: Untrusted application trying to offload to signed PD`. Set `QnnHtpDevice_CustomConfig_t.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD` with `useSignedProcessDomain = false`. ARCH config is silently ignored on real targets (`QnnDsp <W> Specified config ARCH, ignoring on real target`) — the runtime auto-detects from the chip.
- **FastRPC needs a real filesystem path for the Hexagon Skel**. Two prerequisites: (a) set `useLegacyPackaging=true` in AGP `packagingOptions.jniLibs` so the Skel libs extract to `nativeLibraryDir` on install instead of being mmap'd from the APK zip (the DSP can't read APKs); (b) call `setenv("ADSP_LIBRARY_PATH", nativeLibraryDir, 1)` from JNI before any QNN call. Without this, you get `dlopen libQnnHtpV73Stub.so → ... libcdsprpc.so not found` (because the stub also needs `libcdsprpc.so`, see next gotcha).
- **`libcdsprpc.so` / `libadsprpc.so` / `libsdsprpc.so` are vendor public libs that the app's linker namespace excludes by default**. Add `<uses-native-library>` declarations for all three in `AndroidManifest.xml` (under `<application>`), with `android:required="false"` so the manifest still parses on devices without them. Without these, `libQnnHtpV73Stub.so` fails to dlopen the FastRPC client and `QnnDevice_create` returns INVALID_CONFIG with no useful direct error.
- **Enable QNN runtime logging via `QnnLog_create`** before backendCreate/deviceCreate — passes the handle into all create calls. Without it, QNN failures are opaque numeric codes; with it, you get tagged `QnnDsp <E>` / `<W>` / `<I>` messages in logcat that name the actual failure (e.g., "Failed in loading stub: dlopen failed: library libcdsprpc.so not found"). Worth the ~50 LOC of wiring; debugging without it cost an iteration loop or two.
- **xororz UNet I/O dtype is ufp16 (UFIXED_POINT_16, code 0x416) with per-tensor scale/offset quantization, NOT fp16**. The boundary tensors require dequant via `float = (uint16 + offset) * scale` and (for inputs) quant via `uint16 = clamp(round(float/scale) - offset, 0, 65535)`. The metadata reader must read `Qnn_QuantizeParams_t.scaleOffsetEncoding.{scale,offset}` from the binaryInfo templates and the pack/unpack helpers must honor them. Treating ufp16 as fp16 would just hand the DSP/host garbage.
- **xororz UNet graph signature** (verified for `AbsoluteReality_qnn2.28_8gen2.zip`): graph name `model`, inputs `sample` (ufp16 [1,4,64,64]) + `timestamp` (int32 [1]) + `text_embedding` (ufp16 [1,77,768]), output `output` (ufp16 [1,4,64,64]). Slot identification by name substring (`sample`/`timestamp`/`text_embedding`) is robust enough — but the timestep tensor is RAW int32 timestep (not pre-embedded), so just pass the integer t directly.
- **The diffusion probe must run off the main thread**. A full 5-step diffusion takes ~12s total (text-encode + qnn-init + loop). On the main thread that ANRs the app process at startup. The boot probe in `LocalAiApp.kt` uses `kotlin.concurrent.thread(...)` to run probeImagegenNative in the background.

---

## 7. If you need to roll back

Everything image-gen lives in:
- `app/src/main/cpp/` (Plan B — this directory exists for this feature only)
- `app/build/qnn-libs/` (build output, generated by `stageQnnLibs` Gradle task)
- `app/src/main/java/com/nothing/localai/imagegen/` (Kotlin shell)
- AIDL: `generateImage` / `cancelImageGen` + `IImageGenCallback.aidl`
- `LocalAiService.kt`: the `generateImage` binder overrides + `imageGen` field
- `LocalAiApp.kt`: the `nativePing` probe on startup
- `app/build.gradle`: the `externalNativeBuild`, `stageQnnLibs` task, and `jniLibs.srcDirs` additions
- `Aiwidget/.../LocalAiBridge.kt`: ReactMethods + `EVT_IMAGE`
- `Aiwidget/testwidgets/image-gen-1/`
- `scripts/push-diffusion-bundle.sh`

The Gemma 4 LLM path is unaffected.
