# Known issues

## Plan A: MediaPipe `tasks-vision-image-generator` — abandoned

**Status: abandoned. Pivoted to LiteRT (Plan A2) → then to QNN-direct (Plan B).**

### Symptom

Calling `ImageGenerator.createFromOptions(...)` (MediaPipe library, v0.10.26.1)
fails ~12 s into the graph build with:

```
UNKNOWN: Failed to create 2D texture (clCreateImage): Invalid image size
  at third_party/ml_drift/cl/tensor.cc:70
  ...
  at third_party/ml_drift/samples/stable_diffusion/diffuser.cc:497
Calculator::Open() for "StableDiffusionIterateCalculator" failed:
  RET_CHECK failure (mediapipe/.../stable_diffusion_iterate_calculator.cc:253) context_
```

The graph successfully creates Environment → TextGuidance (CLIP) → Copier, then
fails when ml_drift packs the U-Net weight tensors into 2D OpenCL textures.

### What we tried

1. Self-converted SD v1.5 EMA bundle via Google's `image_generator_converter`.
   1040 `.bin` files, 1.9 GB.
2. Protobuf substitution (`protobuf-javalite → protobuf-java:4.26.1`) — fixed a
   `NoSuchMethodError: Any.build()` but didn't help the underlying clCreateImage
   failure.
3. CPU/Vulkan fallback — none exposed by MediaPipe's API.
4. Google-hosted pre-converted bundle — does not exist publicly.

### Diagnosis

ml_drift's CL implementation requests a 2D image larger than the device's
`CL_DEVICE_IMAGE2D_MAX_WIDTH` when packing certain U-Net weight tensors. The
API offers no knob to work around it.

### Coming back to this

Conditions under which a return to MediaPipe Image Generator would be worth re-trying:
- MediaPipe ships a release with the ml_drift CL path tested on Adreno 825 /
  Snapdragon 8s Gen 4.
- Google publishes a pre-converted bundle.
- A `setDelegate(CPU)` or `setDelegate(VULKAN)` lands on `ImageGeneratorOptions.Builder`.

---

## Plan A2: LiteRT + Qualcomm AI Hub TFLite — abandoned

**Status: abandoned. The premise (AI Hub ships pre-built TFLite for SD)
is factually incorrect as of 2026-05.**

### Symptom

Plan A2 assumed we could download a TFLite SD bundle from Qualcomm AI Hub and
load it with `com.google.ai.edge.litert:litert`. Investigation in this session
showed:

1. **Qualcomm's HF (`qualcomm/Stable-Diffusion-v1.5`, `qualcomm/Stable-Diffusion-v2.1`)
   only ships `qnn_context_binary` and `precompiled_qnn_onnx`** — no TFLite
   variant. Both are QNN-based, require Qualcomm's runtime, and are
   pre-compiled for specific Hexagon revisions (none of which include 8s Gen 4).
2. **The `qai-hub-models` Python CLI is argparse-locked to QNN runtimes for SD**:
   ```
   --target-runtime {qnn_context_binary, precompiled_qnn_onnx}
   --precision {w8a16}
   ```
   `tflite` is not an option. The error is: "Model does not support runtime
   tflite with precision w8a16. These combinations are supported:
   w8a16: qnn_context_binary, precompiled_qnn_onnx."
3. **Bypassing the CLI via the `qai_hub` SDK directly** could produce fp16
   TFLite, but only by overriding the model's compile options. The bundled
   `export.py` hardcodes `Precision.w8a16`, which AI Hub then refuses for
   TFLite.
4. **No community SD v2.1 TFLite exists publicly.** Community ports
   (anthrapper, freedomtw, keras-sd) are all SD v1.x via Keras CV — useful as
   a fallback runtime but not for v2.1.
5. **Public SD v2.1 ONNX exists** (`aislamov/stable-diffusion-2-1-base-onnx`,
   fp16). Would require swapping LiteRT → ONNX Runtime Mobile. ~3 GB bundle.
   Estimated ~3–7 min per image on 8s Gen 4 via GPU delegate — outside any
   widget UX budget.

### Why we left

The combination "TFLite + SD v2.1 + no AI Hub account" is empty. The
combination "TFLite + SD v1.5 + community port" is available but tops out at
~30s–2min per image on 8s Gen 4 via GPU delegate, with v1.5-class quality.

### Coming back to this

Worth re-trying when one of these is true:
- AI Hub adds 8s Gen 4 (Hexagon V73) to the supported device list for SD.
- A `--target-runtime tflite` flag becomes available for SD models in
  `qai-hub-models` (track release notes).
- TFLite QNN delegate becomes a maintained path for SD (see
  `quic/ai-hub-apps/apps/image_classification_android/src/main/java/com/qualcomm/qti/sampleapp/tflite_helpers/`
  for the pattern, but no SD reference exists for it).

---

## Plan B: QNN-direct in-tree native build (current)

The active plan. See `IMAGE-GEN-PLAN.md`. Risks/open questions to track here as
they surface:

### Open questions (resolve during implementation)

- **Hexagon revision on 8s Gen 4**: We assume V73 (same as 8 Gen 2) based on
  xororz's `_8gen2.zip` model variant naming. Confirm at Phase 4a by reading
  `QnnHtpDevice_getInfrastructure()` arch info at runtime. If V73 is wrong
  (could be V75/V77), use the `_min.zip` variant or a different bundle.
- **Hexagon Stub/Skel runtime selection**: QNN auto-selects based on device,
  but all `libQnnHtp*.so` Stub + Skel pairs must be packaged. The
  `stageQnnLibs` Gradle task copies V68/V69/V73/V75/V79/V81 by default;
  confirm by checking what files an actual 8s Gen 4 device opens via
  `lsof`-style inspection on a loaded engine.
- **Memory pressure with Gemma 4 E4B loaded**: SD UNet + VAE residency may
  push past available RAM. Phase 6 will reveal whether we need to evict Gemma
  before image gen runs.

### Resolved

- **xororz bundle inner format** (Phase 2, 2026-05-14): plain zip, not
  pickle. Inside the zip is `output_512/qnn_models_8gen2/` containing
  `tokenizer.json`, `clip_v2.mnn`, `pos_emb.bin`, `token_emb.bin`,
  `vae_encoder.bin`, `vae_decoder.bin`, `unet.bin`, and `*.patch` resolution
  overlays. User extracts on the host; push script pushes the directory.
  No in-process zip extraction is needed.
- **Text encoder runtime** (Phase 2, 2026-05-14): xororz packs CLIP as
  `clip_v2.mnn` — an MNN model, not a QNN context binary. MNN is part of
  the primary path, not a CPU fallback as earlier plan revisions suggested.
- **Tokenizer dependency** (Phase 3, 2026-05-14): rather than introducing a
  Rust toolchain (`mlc-ai/tokenizers-cpp` wraps HF Rust tokenizers), we
  hand-rolled CLIP byte-level BPE in `app/src/main/cpp/tokenizer.cpp`.
  Verified bit-exact against Python `tokenizers` on 5 reference prompts.

### Things we already know

- **Local Dream is CC BY-NC** — only their app code. The upstream libs (QNN
  SDK proprietary-but-free-dev, MNN Apache 2.0, tokenizers-cpp Apache 2.0,
  zstd BSD, stb public domain) are independently licensed. We use the upstream
  libs directly without going through Local Dream's source.
- **xororz model files** ship under SD's CreativeML license (downstream of
  CompVis SD 1.5). Personal/hobby use is unambiguous.
