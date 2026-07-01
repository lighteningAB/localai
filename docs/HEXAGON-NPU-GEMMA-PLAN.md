# Fix LiteRT-LM Qualcomm dispatch → run Gemma on Hexagon NPU (SM8735 / V73)

**Goal:** get Gemma 4 E2B running on the Hexagon NPU of a Nothing Phone (SM8735 / Snapdragon 8s Gen 4 / Hexagon **V73**) by fixing the **open-source LiteRT Qualcomm dispatch plugin** (`libLiteRtDispatch_Qualcomm.so`), which is the one fixable link in the chain.

This doc is a complete handoff. A fresh session opened in this repo (`~/Documents/GitHub/LiteRT`) should be able to execute without re-deriving anything. Sibling app repo: `~/Documents/GitHub/localai`.

---

## 1. Why this is the only path to Gemma-on-NPU

Two ways to run an LLM on the Hexagon NPU; for **Gemma** specifically:

| Path | Gemma? | State |
|---|---|---|
| Google **litertlm `Backend.NPU`** (eats the `.litertlm` bundle) | ✅ only Gemma-on-NPU route | ❌ broken — see §2 |
| Qualcomm **AI Hub → Genie** (`genie-t2t-run`, QNN context binaries) | ❌ **no Gemma in catalog** (Llama/Qwen/Phi/Mistral only) | ✅ works, but can't do Gemma |

We have the Qualcomm-specialized model `gemma-4-E2B-it_qualcomm_sm8750.litertlm` (HF `litert-community/gemma-4-E2B-it-litert-lm`, ~2.8GB, on device at `/data/data/com.nothing.localai.debug/files/models/`). Only litertlm loads `.litertlm`. So **fixing litertlm's NPU dispatch is the only route to Gemma on this NPU.** (Genie can't, AI Hub has no Gemma.)

The litertlm runtime layers (`liblitertlm_jni.so`, `libLiteRt.so`) are **closed** (Google-internal build, shipped in the Maven AAR `com.google.ai.edge.litertlm:litertlm-android:0.11.0`). But the QNN **dispatch plugin** they dlopen is **open source in THIS repo** → that's what we build + patch.

---

## 2. Exactly where it fails (verified on device, 2026-06-30)

With everything else solved (dispatch lib present + executable, QNN libs version-matched to QAIRT 2.44, app platform-signed → `platform_app` SELinux domain, model file readable), litertlm `Backend.NPU` init still dies:

```
I litert : Loading shared library: .../libLiteRtDispatch_Qualcomm.so   # dispatch loads fine
W QnnDsp : Initializing HtpProvider
E litert : [dispatch_api.cc:135] Failed to set up QNN manager
E litert : [dispatch_delegate.cc:130] Failed to create a dispatch delegate kernel: No usable Dispatch runtime found
F libc  : Fatal signal 6 (SIGABRT) ... nativeCreateEngine   # uncatchable → DeadObjectException
```

- Fails **host-side, right after `Initializing HtpProvider`, BEFORE any FastRPC/DSP session** → it is **NOT** signed-PD, NOT SELinux, NOT the chip.
- The underlying QNN error is **silenced** (`::qnn::Options LogLevel: 0`; not overridable from the closed AAR).
- **Proof it's the dispatch's QNN config, not the device:** our own `~/Documents/GitHub/localai/app/src/main/cpp/qnn_session.cpp` (Stable Diffusion path) creates a QNN backend + device successfully with the **same** QAIRT 2.44 libs on this **same** SM8735, by explicitly setting **unsigned PD** + an explicit V73 device. litertlm's dispatch does not.

### The smoking-gun line (this repo)
`litert/vendors/qualcomm/dispatch/dispatch_api.cc:130`:
```cpp
if (auto qnn_manager = QnnManager::Create(
        /*options=*/qnn_options,
        /*shared_library_dir=*/shared_library_dir_opt,
        /*soc_model*/ std::nullopt);     // <-- no SoC passed; relies on auto-detect
    ...) {
  ... // on failure: "[dispatch_api.cc:135] Failed to set up QNN manager"
}
```
`QnnManager::Create` impl: `litert/vendors/qualcomm/qnn_manager.cc:681`. Log level plumbing: `litert/vendors/qualcomm/common.h:121`.

**Working hypothesis:** `QnnManager::Create` with `soc_model = std::nullopt` mis-detects / mis-configures the HTP device on SM8735 (soc_model 85, V73), or creates the device without the config our chip needs (cf. unsigned-PD / explicit-arch in `qnn_session.cpp`). Fixing the SoC/device-config in the dispatch should clear it.

---

## 3. The plan

### Phase 0 — go/no-go probe (do this first)
Build the dispatch from source **with QNN logging turned up**, run on device, read the real error behind "Failed to set up QNN manager."
- If the error is in QNN device/backend setup (soc/PD/arch) → **fixable here** (Phase 1).
- If it's an ABI/version handshake with the closed `libLiteRt.so` (`Found Dispatch API with an unsupported version`) → ABI problem (see §5), pivot.

Turn up logging: set `qnn_options` log level to DEBUG/VERBOSE before `QnnManager::Create` in `dispatch_api.cc` (see `common.h:121` `SetLogLevel`), or hard-code it in `QnnManager::Create`. Also add `LITERT_LOG`/`fprintf` around the create call to print the exact `Qnn*` error code.

### Phase 1 — fix the QNN setup
Likely fixes (mirror `qnn_session.cpp`):
1. Pass the real **soc_model = 85** (`QNN_SOC_MODEL_SM8735`) instead of `std::nullopt` in `dispatch_api.cc:130` (or fix the auto-detect in `qnn_manager.cc`).
2. Ensure the HTP **device config** sets **unsigned PD** for untrusted/debug contexts (what `qnn_session.cpp` does via `QnnHtpDevice_CustomConfig` `useSignedProcessDomain=false`). NOTE: on the app side we already platform-sign → `platform_app`, which may make signed-PD OK; but verify.
3. Pass the correct `dsp_arch` (v73) if needed.

### Phase 2 — produce the .so + test
`bazel build --config=android_arm64 //litert/vendors/qualcomm/dispatch:dispatch_api_so` → `libLiteRtDispatch_Qualcomm.so`. Deploy + test (§4).

### Phase 3 — wire into the app
Drop the fixed `.so` into `localai` `app/src/main/jniLibs/arm64-v8a/` (already set up there), rebuild localai (platform-signed, QNN 2.44), select the NPU model in StatusActivity, hit "Test active model". Multimodal (vision/audio on NPU) is a **separate, unproven** question even if text works.

---

## 4. Build + on-device test (the env is already staged)

**Build (Linux + Android NDK + bazel):**
```bash
# in this repo
bazel build --config=android_arm64 \
  //litert/vendors/qualcomm/dispatch:dispatch_api_so
# output: bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so
```
HEAD here is `918c9ca` (2026-06-29). litertlm-android **0.11.0** is ~May 2026 → for the ABI/API-version match you may need to `git fetch --unshallow` and checkout the LiteRT commit/tag matching 0.11.0 (this clone is `--depth 1`).

**Fastest device test = Qualcomm `genie-t2t-run` (runs as `shell`, bypasses app SELinux entirely).** Already staged on device:
- `/data/local/tmp/genie` — QAIRT 2.44 runtime (genie-t2t-run, libGenie, libQnn*.so, V73 skel)
- `/data/local/tmp/genie32` — QAIRT 2.32 runtime
- `/data/local/tmp/genie3b` — imi2 Llama-3.2-3B bundle (context bins + tokenizer + config)

But note: **genie-t2t-run does NOT use libLiteRtDispatch** (Genie talks QNN directly). The dispatch `.so` is only exercised by **litertlm** (`Backend.NPU`). So to test the fixed dispatch you must go through the **localai app** (§3 Phase 3), OR write a tiny C harness that calls the LiteRT Dispatch C API against the gemma `.litertlm`. The app route is wired and easiest.

**localai app NPU test (device side, all prepared):**
```bash
# app: com.nothing.localai.debug (platform-signed, QNN 2.44, dispatch bundled in jniLibs)
adb shell svc power stayon true; adb shell input keyevent KEYCODE_WAKEUP
adb shell am start -n com.nothing.localai.debug/com.nothing.localai.ui.StatusActivity
# tap "...qualcomm_sm8750.litertlm · Hexagon NPU" radio, then "Test active model"
adb logcat | grep -aE 'litert|qnn|NpuDiag|LlmRunner|QnnDsp'
```
Model file on device may need its SELinux MLS category fixed for `platform_app` (one-time, adb is root here):
`adb shell chcon u:object_r:app_data_file:s0:c512,c768 /data/data/com.nothing.localai.debug/files/models/*`

---

## 5. Risks / failure modes (be honest)

1. **ABI/API-version lock (biggest risk).** The closed `liblitertlm_jni.so`/`libLiteRt.so` call the dispatch through a versioned C API (`LiteRtDispatchGetApi`). A from-HEAD build may report `Found Dispatch API with an unsupported version` and not load (upstream google-ai-edge/LiteRT **#6889**, LiteRT-LM **#1121**). Mitigation: match the LiteRT commit to litertlm-android 0.11.0.
2. **Bug may be upstream of the dispatch** — inside closed `libLiteRt.so`/`liblitertlm_jni.so`. "Failed to set up QNN manager" being in the **open** `dispatch_api.cc`/`qnn_manager.cc` is a good sign it's reachable, but not a guarantee.
3. **Multimodal** (vision/audio encoders on NPU) is unproven even if text-decode works. Best case from this effort is likely **Gemma text on NPU**.
4. Effort is real (bazel build + QNN debugging + ABI matching). GPU already runs Gemma 4 E2B **multimodal** today (localai toggle) — that's the fallback.

---

## 6. Key facts / reference

- **Device:** SM8735 "sun" / Snapdragon 8s Gen 4 / **Hexagon V73**. `QNN_SOC_MODEL_SM8735 = 85`. user build, SELinux Enforcing, but **`adb root` works** (can chcon/restorecon/read tombstones).
- **QAIRT:** we have full SDKs extracted in `localai` scratchpad: 2.44 (`~/Downloads/qairt/2.46.0.260424` is also present), 2.32 + 2.44 under `/private/tmp/claude-501/.../scratchpad/npulibs/` (these are session-temp; re-download from `softwarecenter.qualcomm.com/.../Qualcomm_AI_Runtime_Community/All/<ver>/v<ver>.zip` if gone). AI Hub compiles LLMs with **QAIRT 2.42**.
- **Reference QNN-on-SM8735 code that WORKS:** `~/Documents/GitHub/localai/app/src/main/cpp/qnn_session.cpp` — does QnnDevice_create with **unsigned PD** + explicit V73. Copy its device-config approach into the dispatch.
- **localai NPU wiring (done):** `ModelId.GEMMA4_E2B_NPU`, `ModelPrefs`, `LlmRunner` (NPU branch: ADSP path + `Backend.NPU(dispatchDir)` auto-detect), AIDL `get/setActiveModel`, StatusActivity radio + "Test active model" button, `app/src/main/jniLibs/arm64-v8a/libLiteRtDispatch_Qualcomm.so` (the prebuilt — replace with our fixed build), server-sign `apkSignKey "platform"`. See `localai/README.md` "Hexagon NPU bundle" + memory `project_localai_npu_llm`.
- **Upstream issues:** google-ai-edge/LiteRT #6889 (publish prebuilt dispatch), LiteRT-LM #1121, #1377, #2226 (Qualcomm NPU broadly not-working). GDE article "Bringing Multimodal Gemma 4 E2B to the Edge" fell back to GPU.

## 7. First action for the new session
1. Set up bazel + Android NDK for `--config=android_arm64` in this repo (check `.bazelrc` / `WORKSPACE`).
2. Patch `dispatch_api.cc` / `qnn_manager.cc` to (a) raise QNN log level, (b) print the real error around `QnnManager::Create`.
3. Build `//litert/vendors/qualcomm/dispatch:dispatch_api_so`, drop into `localai/app/src/main/jniLibs/arm64-v8a/`, rebuild+install localai, run the StatusActivity NPU test, read the real error.
4. Decide fixable-here vs ABI/closed-lib dead-end from that output.

---

## 8. RESULTS — session 2 (2026-06-30, verified on device A024 / SM8735)

**The dispatch chain is now fully repaired and verified on-device. The remaining blocker is the model artifact's arch, not the dispatch.**

### Fixes that work (validated by logcat)
1. **QNN System-version gate.** HEAD pins `@qairt` = QAIRT **2.47** (System API 1.11). The litertlm-android **0.11.0** AAR bundles the QNN runtime libs (libQnnSystem.so etc.) at System API **1.8**. `qnn_manager.cc:331/342` rejects runtime minor < compiled minor → `[qnn_manager.cc:350] Qnn System library version 1.8.0 is mismatched. minimum 1.11.0` → "Failed to set up QNN manager". **Fix:** build the dispatch against **QAIRT 2.44.0.260225** (System API 1.8, matches the AAR libs). Done via `LITERT_QAIRT_SDK=/Users/patrickfan/Downloads/` with `~/Downloads/qairt/2.47.0.260601` symlinked → `2.44.0.260225` (reuses the hardcoded strip_prefix; no workspace.bzl edit). 2.44 SDK stable copy at `~/Downloads/qairt/2.44.0.260225`.
2. **SoC detection.** On-device `deviceGetPlatformInfo` returns `socModel = 91` for this SM8735 ("sun") — NOT 85 (QAIRT QnnTypes.h `QNN_SOC_MODEL_SM8735 = 85` is wrong for this silicon; 91 is absent from the enum, QNN HTP still prints "Detected SM8735"). `FindSocInfo(91)` missed → `kSocInfos[0]` = UNKNOWN/NONE → `htp_backend.cc:427 "SoC info was not configured successfully"`. **Fix:** added `SnapdragonModel::SM8735_SUN = 91` (soc_table.h) + `SocInfo("SM8735", SM8735_SUN, V73, 8)` (soc_table.cc). (Also added SM8735=85 for other units.) After fix: `platform_info OK; raw socModel=91 matched=1 resolved soc=SM8735 arch=73`. QnnManager::Create, backendCreate, deviceCreate all SUCCEED.
3. **ABI (§5) did not bite.** Closed libLiteRt.so 0.11.0 accepted the HEAD-built dispatch (no "unsupported version").

### Current wall (NOT fixable in the dispatch)
- After init succeeds, loading the precompiled NPU graph fails: `QnnDsp <E> Failed to register context to device and backend` / `Failed to create context from binary with err 0x138d` = **5005 = QNN_CONTEXT_ERROR_CREATE_FROM_BINARY** (`qnn_manager.cc:671`, `litert_dispatch_invocation_context.cc:239`).
- Reproduces under `setenforce 0` (permissive) → **not SELinux**. (FastRPC AVC denials for platform_app→/dev/fastrpc-cdsp appear but are non-fatal.)
- Cause: the model `gemma-4-E2B-it_qualcomm_**sm8750**.litertlm` ships a context binary compiled for **SM8750 / Hexagon V79**. Device is **SM8735 / V73**. QNN HTP context binaries are arch-locked; V79 blob can't register on V73. Confirmed: device has libQnnHtpV73Skel.so (can run V73), partition_0 is the 1.18GB prefill_decode section.
- No published V73 Gemma bundle exists (ModelId.kt: only sm8750/qcs8275; qcs8275=V75). **To run Gemma on SM8735 NPU you must recompile the model to a V73 QNN context binary** via the LiteRT qnn_compiler_plugin targeting SM8735 — or use the GPU path (already works, multimodal).

### Debug patches still in the dispatch source (revert for a clean build)
- `dispatch_api.cc`: forced `qnn_options.SetLogLevel(kDebug)` + `QNNLogger::SetLogLevel(kDebug)` + `[NPU-DEBUG]` logs around QnnManager::Create.
- `htp_backend.cc`: `[NPU-DEBUG]` QNN_LOG_ERROR printing numHwDevices/raw socModel/match/resolved arch.
- Keep: soc_table.h/.cc SM8735 entries. Keep: QAIRT 2.44 pin for matching litertlm 0.11.0.

### localai-side notes
- `app/build.gradle`: added `androidResources { noCompress += ['.bin','.tflite','.litertlm'] }` (the 2.5GB sdxl_lightning_unet.bin overflowed aapt's compressor: "Required array size too large").
- Temporarily moved `app/src/main/assets/sdxl_lightning_unet.bin` → `localai/_assets_stash/` to build (SD asset, unrelated to Gemma-NPU). **Restore it** for image-gen.
