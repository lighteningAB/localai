# Runbook — AOT-compile Gemma 4 E2B for the SM8735 (V73) NPU

**Goal:** produce a `.litertlm` whose QNN context binary is compiled for **Hexagon V73**, so it loads on the Nothing Phone SM8735 ("sun") NPU via the already-fixed `libLiteRtDispatch_Qualcomm.so`.

**Key trick:** the AOT toolchain's SoC list (`litert/python/aot/vendors/qualcomm/target.py`) has **no SM8735** entry, but it has **SM8550 = V73**. HTP context binaries are keyed on *arch* (V73), not exact SoC — so compile for **SM8550** and the V73 binary runs on SM8735. (Proven: localai's Stable-Diffusion path uses an SM8550/V73 binary on this SM8735 device today.)

> Runs on **x86_64 Linux only** — the AOT step uses `libQnnHtpPrepare.so` (x86-64 Linux). A macOS/arm64 box cannot do this. Use a Linux workstation or cloud VM.

---

## Prerequisites (on the x86_64 Linux host)

1. Python 3.11+ venv.
2. `pip install ai-edge-litert[npu-sdk] litert-torch transformers torch` (the `[npu-sdk]` extra pulls the QNN/QAIRT AOT pieces).
3. **QAIRT SDK** matching the device runtime: the device runs litertlm-android 0.11.0 whose QNN libs are **System API 1.8 (QAIRT 2.44.0.260225)**. Use QAIRT **2.42–2.44** for the compile so the produced context binary is loadable by the 1.8 runtime. Set `QAIRT_ROOT=~/qairt/2.44.0.260225` (or 2.42). Do NOT use 2.46/2.47 (System API 1.10/1.11 → newer-than-runtime risk).
4. HF access to the gated `google/gemma-4-E2B-it` checkpoint (`huggingface-cli login`; you already have litert-community access).

## Compile (Route 1 — all-in-one `export-hf`, recommended)

```bash
export QAIRT_ROOT=~/qairt/2.44.0.260225           # match device runtime (System API 1.8)
export HF_TOKEN=hf_xxx                              # gated gemma-4 access

litert-torch export-hf \
  --model=google/gemma-4-E2B-it \
  --output_dir=/tmp/gemma4-e2b-sm8550-v73 \
  --split_cache \
  --externalize_embedder \
  --quantization_recipe=''            \  # NPU = full-float or SRQ ONLY (weight-only-int4 unsupported). '' = full float.
  --aot_backend=qualcomm \
  --aot_soc_model=SM8550                 # V73 — same arch as SM8735
# → /tmp/gemma4-e2b-sm8550-v73/<name>.litertlm
```

Notes / likely adjustments:
- **Quantization:** `''` (full float) makes a large bundle (~5GB) and is slow. The shipped sm8750 bundle (2.58GB) is SRQ. If full-float is too big, use the SRQ recipe (needs a small calibration set) — check `litert-torch export-hf --help` for the qualcomm SRQ recipe flag. Start with `''` to get *something* loading, optimize quant after.
- **Multimodal:** vision/audio encoder NPU export for gemma-4-E2B is WIP (litert-torch issue #1039). Expect **text decoder on NPU**; keep vision/audio on GPU/CPU (localai already does — `visionBackend = Backend.GPU()`).
- **Route 2 (manual, more control, more failure modes):** `ai_edge_torch` (PyTorch→tflite) → `apply_plugin_main --soc_model SM8550 --soc_manufacturer Qualcomm` (tflite→NPU tflite) → `litertlm_builder_cli` (→.litertlm). Issue #960 hit `Failed to create QNN context: 1002 / unresolved custom op DISPATCH_OP` on this path — prefer Route 1.

## Deploy + run (on the Mac / device side — dispatch already fixed)

```bash
export ANDROID_SERIAL=2312M154M000068
adb push /tmp/gemma4-e2b-sm8550-v73/<name>.litertlm /data/local/tmp/
adb shell run-as com.nothing.localai.debug cp /data/local/tmp/<name>.litertlm files/models/
# fix SELinux MLS category so platform_app can read it:
adb shell chcon u:object_r:app_data_file:s0:c512,c768 \
  /data/data/com.nothing.localai.debug/files/models/<name>.litertlm
```
- Repoint the NPU model: in `localai/app/src/main/java/com/nothing/localai/llm/ModelId.kt`, set `GEMMA4_E2B_NPU.fileName` to `<name>.litertlm` (and `supportsVision/Audio=false` until multimodal-on-NPU is proven). Rebuild + install localai.
- The runtime dispatch is already built (QAIRT 2.44 + socModel 91→V73 in soc_table) and deployed. StatusActivity → NPU radio → "Test active model".

## What to watch for next (expected once a V73 binary loads)
- **contextCreateFromBinary** should now SUCCEED (arch match V73↔V73) instead of `0x138d`.
- **Possible next blocker = unsigned PD.** localai's `qnn_session.cpp` sets `QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD useSignedProcessDomain=false`; the litert dispatch's arm64 path does NOT (that block is `#if __x86_64__`). If you hit "Untrusted application trying to offload to signed PD" / a FastRPC failure at context-register, add an unsigned-PD `QnnDevice_CustomConfig` in `htp_backend.cc` `Init` for the arm64 path, mirroring `qnn_session.cpp`. (We couldn't test PD before because the V79 binary failed on arch first.)

## Why this is the only route for Gemma 4
No published V73 Gemma-4 bundle exists — `litert-community/gemma-4-E2B-it-litert-lm` ships only `sm8750` (V79) + `qcs8275` (V75). Gemma**3**-1B has an SM8550/V73 build (would run now) but you want Gemma 4. So Gemma 4 must be AOT-compiled for V73 yourself via the above.
