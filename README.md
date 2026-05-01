# LocalAi

Sibling APK that hosts on-device models and exposes them to Aiwidget over AIDL.

## Status

Default model: **Gemma 4 E4B IT (int4, ~2.96 GiB)** — multimodal text + image
(audio modality wired in the AIDL but disabled in the runner until the
`setAudioModelOptions` config is sorted). Gemma 3n E2B/E4B and Gemma 4 E2B are
also in the catalog and can be selected via `ModelId.DEFAULT`.

Standalone vision (`classifyImage`) and audio (`transcribe`, `speak`) AIDL
methods are stubbed — multimodal input goes through `addImage` / `addAudio`
on a chat session.

## Toolchain

- AGP 8.9.1 / Kotlin 2.1.10 / Gradle 8.12 (matches Aiwidget)
- minSdk 31, target 36
- arm64-v8a only
- MediaPipe `tasks-genai:0.10.28`

## Model setup (Gemma 4 E4B IT, int4)

Until a download URL is wired into `ModelCatalog`, push manually:

```bash
# 1. Sign in to HuggingFace, accept the Gemma license, then download
#    https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
#    The file is gemma-4-E4B-it-web.task (~2.96 GB).

# 2. Push via the helper script:
~/Documents/GitHub/localai/scripts/push-model.sh ~/Downloads/gemma-4-E4B-it-web.task

# Or manually:
adb push gemma-4-E4B-it-web.task /data/local/tmp/
adb shell "run-as com.nothing.localai.debug mkdir -p files/models"
adb shell "run-as com.nothing.localai.debug cp /data/local/tmp/gemma-4-E4B-it-web.task files/models/"
```

For the smaller **E2B** variant (~2.0 GiB, faster cold load, lower quality), push
`gemma-4-E2B-it-web.task` from `litert-community/gemma-4-E2B-it-litert-lm` and
flip `ModelId.DEFAULT` to `GEMMA4_E2B_INT4`.

For the older **Gemma 3n** family (still in the catalog), download from
`huggingface.co/google/gemma-3n-{E2B,E4B}-it-litert-preview` and use the
`gemma-3n-{E2B,E4B}-it-int4.task` filenames.

## Signing

For `signature`-level binding to work, this APK and Aiwidget must be signed
with the same key. Debug builds share the Android debug key automatically.
For dev convenience, `BIND_AI` is currently `protectionLevel="normal"`.

## Modalities

- Text: `addQueryChunk` (handled internally by `generate(prompt)`)
- Image: `addImage(jpegFd)` — JPEG decoded to Bitmap; max 4 images per turn
- Audio: `addAudio(pcmFd, 16000)` — 16 kHz mono PCM16; caller resamples

Audio modality is currently disabled in `LlmRunner.newSession()` because
`setEnableAudioModality(true)` alone causes the inference graph to fail with
"audio options should not be null". Re-enable when wiring the mic widget by
also providing the appropriate `AudioModelOptions` config.

## Hardware notes

GPU delegate (Adreno via OpenCL) is the default acceleration path for the
universal `-web.task` variant. Snapdragon NPU acceleration would require
switching to LiteRT-LM with the QNN HTP delegate and a chip-specific
`.litertlm` model — separate runtime, not wired here.
