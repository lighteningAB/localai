# LocalAi

Sibling APK that hosts on-device models and exposes them to Aiwidget over AIDL.

## Status

> **Branch `experiment/litertlm-0.11.0-gemma4-e4b`** — validated end-to-end on
> Snapdragon 8s Gen 4 (SM8735). All three modalities working on Gemma 4 E4B:
> text + image (GPU/OpenCL) + audio (CPU/XNNPACK, PCM16 wrapped in WAV). The
> 0.10.2 vision SIGSEGV that drove the previous revert is gone in 0.11.0.
> Ready to fold back into `main` when convenient.

Default model: **Gemma 4 E4B IT (`.litertlm`, ~3.66 GiB)** — multimodal text +
image + audio. Gemma 3n E2B/E4B `.task` specs are still in the catalog but the
runner only loads `.litertlm` now; selecting a `.task` model id will fail at
load time. Gemma 4 E2B is also in the catalog (~2.59 GB) and selectable via
`ModelId.DEFAULT = GEMMA4_E2B_INT4`.

Standalone vision (`classifyImage`) and audio (`transcribe`, `speak`) AIDL
methods are stubbed — multimodal input goes through `addImage` / `addAudio`
on a chat session.

## Toolchain

- AGP 8.9.1 / Kotlin 2.3.0 / Gradle 8.12
- minSdk 31, target 36
- arm64-v8a only
- LiteRT-LM `com.google.ai.edge.litertlm:litertlm-android:0.11.0`

Kotlin 2.3.0 is required for this branch: `litertlm-android:0.11.0` ships
Kotlin metadata 2.3.0 and won't read on 2.1.x. Aiwidget can stay on 2.1.10 —
AIDL is generated Java, so the cross-APK contract is unaffected by the
Kotlin version drift.

## Model setup (Gemma 4 E4B IT, `.litertlm`)

Until a download URL is wired into `ModelCatalog`, push manually:

```bash
# 1. Sign in to HuggingFace, accept the Gemma license, then download the
#    generic .litertlm (NOT the -web.task; that's a browser/WebGPU bundle and
#    won't load in LiteRT-LM-Android):
#    https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
#    File: gemma-4-E4B-it.litertlm  (~3.66 GB)

# 2. Push via the helper script:
~/Documents/GitHub/localai/scripts/push-model.sh ~/Downloads/gemma-4-E4B-it.litertlm

# Or manually:
adb push gemma-4-E4B-it.litertlm /data/local/tmp/
adb shell "run-as com.nothing.localai.debug mkdir -p files/models"
adb shell "run-as com.nothing.localai.debug cp /data/local/tmp/gemma-4-E4B-it.litertlm files/models/"
```

For the smaller **E2B** variant (~2.59 GB, faster cold load, lower quality), push
`gemma-4-E2B-it.litertlm` from `litert-community/gemma-4-E2B-it-litert-lm` and
flip `ModelId.DEFAULT` to `GEMMA4_E2B_INT4`. (The E2B repo also ships
`qualcomm_sm8750` and `qualcomm_qcs8275` NPU-specialized bundles; neither
matches our SM8735 / Snapdragon 8s Gen 4, so we stay on the generic bundle.)

## Modalities

- **Text** — `addQueryChunk` (handled internally by `generate(prompt)`).
  Decoder runs on XNNPACK CPU.
- **Image** — `addImage(jpegFd)`. JPEG PFD staged to `cacheDir/session-<sid>/img-*.jpg`,
  fed as `Content.ImageFile(path)`. Max 4 images per turn. Vision encoder runs
  on OpenCL via the `LITERT_CL` delegate (Adreno).
- **Audio** — `addAudio(pcmFd, sampleRate)`. PCM16 mono PFD read into memory,
  wrapped in a 44-byte WAV header (because `Content.AudioBytes` runs through
  miniaudio, which won't decode raw PCM), and fed as `Content.AudioBytes(byte[])`.
  Gemma 4's USM encoder expects 16 kHz; a different rate is logged but not
  rejected. Audio encoder + adapter run on XNNPACK CPU — the bundle's audio
  weights are CPU-only and `Engine.initialize()` rejects GPU.

## Session model (single-active)

LiteRT-LM 0.11.0 enforces **one `Conversation` per `Engine`** — the release
notes' "multi-session support" is CLI-only; the Android binding throws
`FAILED_PRECONDITION: A session already exists` on the second
`createConversation`. `SessionRegistry` enforces this by closing every other
live `ChatSession` before constructing a new one. Multi-widget UX still works,
just serialized: whichever widget is most recently touched holds the live
conversation; stale widgets rebuild KV cache on their next turn.

## Smoke test

```bash
# 1. Install debug APK
./gradlew :app:installDebug

# 2. Tail logcat in another shell, scoped to the tags that matter
adb logcat -c && adb logcat \
  LlmRunner:V LocalAiService:V SessionRegistry:V \
  tflite:V AndroidRuntime:E libc:F DEBUG:F "*:S" \
  | grep -v "XNNPack weight cache: written"

# 3. From the Aiwidget side, exercise text / image / audio via the
#    chatbot-1, camera-vision-1, audio-prompt-1 widgets.
```

What you should see on a healthy cold load (one-time; subsequent loads hit
the on-disk XNNPack weight caches):

```
tflite:  Initialized TensorFlow Lite runtime.
tflite:  XNNPack weight cache loaded from ...gemma-4-E4B-it.litertlm.xnnpack_cache_*
tflite:  Replacing 2199 out of 2712 nodes with delegate (TfLiteXNNPackDelegate)   # text decoder
tflite:  Replacing 8 out of 8 nodes ...                                           # MTP heads (x16)
tflite:  Loaded OpenCL library with dlopen.
tflite:  Replacing 1477 out of 1477 nodes with delegate (LITERT_CL)               # vision encoder (x3 subgraphs)
tflite:  XNNPack weight cache loaded ...vision_adapter.xnnpack_cache_*
tflite:  XNNPack weight cache loaded ...static_audio_encoder.xnnpack_cache_*
tflite:  XNNPack weight cache loaded ...audio_adapter.xnnpack_cache_*
```

If you instead see `Fatal signal 11 (SIGSEGV)` referencing
`liblitertlm_jni.so`, the vision path regressed — file an issue against
[google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/issues)
with the chip (SM8735), LiteRT-LM version, model bundle, and crashing offset.

## Signing

For `signature`-level binding to work, this APK and Aiwidget must be signed
with the same key. Debug builds share the Android debug key automatically.
For dev convenience, `BIND_AI` is currently `protectionLevel="normal"`.

## Hardware notes

LiteRT-LM 0.11.0 runs the text decoder on XNNPACK CPU, the vision encoder
on Adreno via OpenCL (`LITERT_CL` delegate), and the audio encoder on
XNNPACK CPU. No chip-specialized `.litertlm` exists for SM8735 (Snapdragon
8s Gen 4) on the HuggingFace `litert-community` repos — only
`qualcomm_sm8750` (8 Elite) and `qualcomm_qcs8275` (Dragonwing IoT). Even on
this runtime we don't get Hexagon NPU acceleration; the win versus MediaPipe
`tasks-genai` is Gemma 4 quality + full multimodal coverage, not raw
throughput.

If/when Google ships a `sm8735` bundle, swap the filename in `ModelCatalog`
and the runner will pick up NPU automatically — no other changes needed.
