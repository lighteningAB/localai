# LocalAi

Sibling APK that hosts on-device models and exposes them to Aiwidget over AIDL.

## Status

> **Experiment branch — `experiment/litertlm-0.11.0-gemma4-e4b`.**
> Swapped the runtime back from MediaPipe `tasks-genai` to LiteRT-LM 0.11.0
> to re-test Gemma 4 E4B multimodal vision on Snapdragon 8s Gen 4 (the 0.10.2
> SIGSEGV that drove the previous revert). Run the smoke test below; if it
> crashes again, `git checkout main` reverts cleanly.

Default model: **Gemma 4 E4B IT (`.litertlm`, ~3.66 GiB)** — multimodal text +
image. Audio modality is *not* wired in this experiment branch (vision-only
isolation). Gemma 3n E2B/E4B `.task` specs are still in the catalog but the
runner only loads `.litertlm` now; selecting a `.task` model id will fail at
load time.

Standalone vision (`classifyImage`) and audio (`transcribe`, `speak`) AIDL
methods are stubbed — multimodal input goes through `addImage` / `addAudio`
on a chat session.

## Toolchain

- AGP 8.9.1 / Kotlin 2.1.10 / Gradle 8.12 (matches Aiwidget)
- minSdk 31, target 36
- arm64-v8a only
- LiteRT-LM `com.google.ai.edge.litertlm:litertlm-android:0.11.0`

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

## Smoke test (vision SIGSEGV repro)

After pushing the model, fire one image+text turn end-to-end:

```bash
# 1. Install debug APK
./gradlew :app:installDebug

# 2. Tail logcat in another shell
adb logcat -c && adb logcat LlmRunner:V LocalAiService:V *:E

# 3. From the Aiwidget side (or any test harness binding to the AIDL),
#    call addImage(jpegFd) then generate(sid, "describe this image", cb).
#    Cold load takes ~10s for the 3.66 GB .litertlm; first token latency
#    after that should be sub-second on Adreno via the GPU backend.
```

What you're looking for in logcat:
- `Fatal signal 11 (SIGSEGV)` referencing `liblitertlm_jni.so` → vision path
  still broken on 0.11.0, file an issue against
  [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/issues)
  with the chip (SM8735), LiteRT-LM version, model bundle, and offset.
- Clean token stream → 0.11.0 fixed it silently; consider folding this branch
  back into `main` and retiring the Gemma 3n `.task` path.

## Signing

For `signature`-level binding to work, this APK and Aiwidget must be signed
with the same key. Debug builds share the Android debug key automatically.
For dev convenience, `BIND_AI` is currently `protectionLevel="normal"`.

## Modalities

- Text: `addQueryChunk` (handled internally by `generate(prompt)`)
- Image: `addImage(jpegFd)` — JPEG decoded to Bitmap; max 4 images per turn
- Audio: `addAudio(pcmFd, 16000)` — 16 kHz mono PCM16; caller resamples

Audio modality is not wired in this experiment branch. The Gemma 4 `.litertlm`
bundle is expected to support audio per the model card, but the public
LiteRT-LM Android quickstart only documents `EngineConfig.visionBackend`;
audio backend config is a follow-up after the vision question is settled.

## Hardware notes

LiteRT-LM 0.11.0 GPU backend via Adreno (OpenCL) is the acceleration path
here. No chip-specialized `.litertlm` exists for SM8735 (Snapdragon 8s Gen 4)
on the HuggingFace `litert-community` repos — only `qualcomm_sm8750` (8 Elite)
and `qualcomm_qcs8275` (Dragonwing IoT). So even on this runtime we don't get
NPU acceleration; the win versus MediaPipe `tasks-genai` would purely be
Gemma 4 quality + multimodal coverage, not throughput.
