# Gemma on Hexagon NPU — state review + forward plan (2026-07-02)

Successor to `GEMMA_NPU_DISPATCH_FIX.md` (diagnosis, written 2026-06-30) and
`GEMMA4_V73_AOT_RUNBOOK.md` (AOT compile recipe). This doc reviews everything done so
far, calls out the mistakes found on review, and lays out the corrected path to
tokens-on-NPU. Read this first; dip into the other two docs for detail.

---

## 1. Where we actually are

The original blocker chain is **beaten**. The rebuilt dispatch works. What remains is a
**model-artifact problem**, not a runtime problem.

| Layer | State | Evidence |
|---|---|---|
| localai NPU toggle (ModelId/ModelPrefs/LlmRunner/AIDL/StatusActivity) | ✅ done, committed | localai branch `experiment/hexagon-npu-gemma` (`f28a227`, pushed) |
| QNN runtime on device (QAIRT 2.44, V73 skel) | ✅ works | SD imagegen + genie-t2t-run both init QNN/HTP on this SM8735 |
| App signing / SELinux (`platform` key → `platform_app`) | ✅ solved | server-sign `apkSignKey "platform"`; exec-from-jniLibs; MLS `chcon` for pushed files |
| litertlm dispatch (`libLiteRtDispatch_Qualcomm.so`) | ✅ **FIXED, rebuilt from this repo** | patches in working tree (see §3); rebuilt .so = 674688 B, in localai `jniLibs/` |
| Root cause of "Failed to set up QNN manager" | ✅ found | device reports **fused socModel 91** (a "sun"/SM8735 variant absent from QAIRT's enum *and* litert's soc table) → `FindSocInfo` missed → bad fallback. Patched table: 85 & 91 → V73 |
| Loading `gemma-4-E2B-it_qualcomm_sm8750.litertlm` | ❌ **impossible on this device** | its QNN context binary is compiled for **V79**; our DSP is **V73**. `contextCreateFromBinary` → `0x138d` arch mismatch. No fix can make a V79 binary run on V73 |
| A Gemma-4 `.litertlm` with a **V73** context binary | ❌ doesn't exist publicly | `litert-community` ships only `sm8750` (V79) and `qcs8275` (V75). Must AOT-compile ourselves → `GEMMA4_V73_AOT_RUNBOOK.md` |

**Net:** runtime stack ✅ end-to-end up to context-load; we now need a V73 model artifact.

---

## 2. Issues found reviewing the prior sessions' work

1. **The "HTP forward-compatible" assumption was backwards.** Earlier sessions assumed
   the `sm8750` bundle would run on our chip. HTP compatibility runs
   *older-arch binary → newer-arch chip only*. A V79 binary can never run on a V73 DSP.
   The SD precedent that seeded this belief was a ≤V73 binary on V73. Cost: the whole
   "bundle-first" test could never have produced tokens, even with a perfect dispatch.
   (The dispatch fixes it forced were real and necessary, so not wasted — but the
   reasoning was wrong and the docs asserted it confidently.)
2. **`ModelId.GEMMA4_E2B_NPU` still points at the sm8750 file** — a bundle that cannot
   work on this device. Must be repointed to the future V73 artifact (and
   `supportsVision/Audio=false` until multimodal-on-NPU is proven).
3. **Recovery risk (was critical, fixed as of this doc):** the LiteRT patches
   (soc_table 85/91, htp_backend debug logs, dispatch_api forced-kDebug) and the
   rebuilt .so existed only as uncommitted local files. Now preserved: patch file +
   rebuilt .so committed to localai `experiment/hexagon-npu-gemma` and pushed
   (`docs/patches/litert-dispatch-sm8735.patch`, `app/src/main/jniLibs/`).
4. **The v2.1.5 prebuilt's backup (`*.so.bak.*`) was deleted** during a cleanup pass.
   Harmless — re-extractable from the LiteRT **v2.1.5** release asset
   `litert_npu_runtime_libraries.zip` (`qualcomm_runtime_v73/`).
5. **Two cheap validation assets were never used:**
   - **`litert_lm_main.android_arm64`** — the LiteRT-LM **v0.11.0 release CLI**
     (same version family as the app's AAR). Runs as `shell`, prints stderr directly,
     needs no APK/signing/SELinux dance. This is the fastest iteration loop for every
     NPU experiment and should be the default harness (§4 step 1).
   - **Gemma3-1B has a published V73 bundle** (`Gemma3-1B-IT_q4_ekv1280_SM8550.litertlm`,
     ~657 MB, litert-community). SM8550 = V73 = our arch. It can validate the entire
     rebuilt-dispatch → QNN → DSP path **today**, before any AOT work.
6. **Unverified claims to re-check before relying on them** (flagged, not disproven):
   the exact `litert-torch export-hf` flags in the AOT runbook (`--aot_soc_model`,
   `--split_cache`, `--externalize_embedder`, quant recipe names), the cited issue
   numbers (#1039/#960), and the `ai-edge-litert[npu-sdk]` extra name. Verify against
   `--help` on the Linux box before running the big export.
7. **VTCM guess:** the soc-table patch gives SM8735 `vtcm_size_in_mb = 8` (copied from
   SM8550/SM8750). Plausible but unverified — the `[NPU-DEBUG]` platform-info log line
   prints what the device actually reports; check it on the next run.
8. **State drift risks:** (a) it's unverified whether the currently-installed APK
   contains the *rebuilt* (674688 B) dispatch — device was offline at review time;
   rebuild + `adb install -r` before the next test. (b) **Never `adb uninstall`
   localai** — it defines `BIND_AI`, and uninstalling wiped WidgetForge's grant once
   already (fixed by reinstalling widgetforge). Always `install -r`. (c) The device's
   SD bundle `files/models/sd-v15-xororz` was wiped during the signing churn — re-push
   if imagegen is needed.
9. **Stale docs/memory:** `GEMMA_NPU_DISPATCH_FIX.md` §2/§3 predate the socModel-91
   discovery and the arch-mismatch finding; localai `docs/HEXAGON-NPU-GEMMA-PLAN.md`
   (committed copy) likewise. This doc supersedes both; memory updated 2026-07-02.

---

## 3. The LiteRT patches (this repo, base `918c9ca`)

- `litert/vendors/qualcomm/core/schema/soc_table.h` — enum: `SM8735 = 85`,
  `SM8735_SUN = 91` (fused socModel some sun units report; absent from QAIRT enum).
- `litert/vendors/qualcomm/core/schema/soc_table.cc` — table rows: both → `DspArch::V73`, 8 MB VTCM.
- `litert/vendors/qualcomm/core/backends/htp_backend.cc` — `[NPU-DEBUG]` platform-info
  logging (raw socModel, matched?, resolved arch).
- `litert/vendors/qualcomm/dispatch/dispatch_api.cc` — force QNN log level `kDebug`
  (litertlm passes log-off, which is what hid the real error for two days).

Build: `bazel build --config=android_arm64 //litert/vendors/qualcomm/dispatch:dispatch_api_so`
→ `bazel-bin/.../libLiteRtDispatch_Qualcomm.so` (674688 B). Copy of the diff:
`localai/docs/patches/litert-dispatch-sm8735.patch` (committed + pushed).

Upstreaming the soc-table rows (85/91) to google-ai-edge/LiteRT is worth a small PR —
it would fix every SM8735 device and is uncontroversial.

---

## 4. Forward plan — **the goal is Gemma 4.** Critical path = the AOT compile (step 3);
start it immediately (Linux box setup + flag verification have no dependency on
anything else). Steps 1–2 are *optional de-risk probes* that can run in parallel —
they do NOT gate step 3. Their only purpose: if the Gemma-4 artifact later fails to
load, a passed probe proves the runtime stack is fine and isolates the fault to our
compile. If you'd rather go straight at Gemma 4, skip to step 3 and fall back to
step 1 only on failure.

### Step 1 (optional, parallel) — validate the runtime with the published V73 Gemma3-1B (½ day)
Diagnostic only — Gemma 3 is NOT the goal. It's the only *known-good* V73 `.litertlm`
in existence, from the same litert-community pipeline, so it exercises the identical
litertlm→dispatch→QNN→V73 path our Gemma-4 artifact will use.
No AOT, no app. Download `Gemma3-1B-IT_q4_ekv1280_SM8550.litertlm` (litert-community,
gated HF — same account access as gemma-4) and the **LiteRT-LM v0.11.0 CLI**
(`litert_lm_main.android_arm64` release asset). Stage a self-contained dir:

```bash
adb shell mkdir -p /data/local/tmp/lm
adb push litert_lm_main.android_arm64 /data/local/tmp/lm/
adb push Gemma3-1B-IT_q4_ekv1280_SM8550.litertlm /data/local/tmp/lm/
# rebuilt dispatch + QAIRT 2.44 QNN libs + V73 skel (reuse /data/local/tmp/genie/*.so)
cp bazel-bin/.../libLiteRtDispatch_Qualcomm.so → push to /data/local/tmp/lm/
adb shell 'cd /data/local/tmp/lm && chmod 755 litert_lm_main.android_arm64 && \
  LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD ./litert_lm_main.android_arm64 \
  --model_path=Gemma3-1B-IT_q4_ekv1280_SM8550.litertlm --backend=npu \
  --input_prompt="What is the capital of France?"'   # check --help for exact flags
```
- **Tokens appear → the entire NPU-LLM stack incl. our dispatch fix is proven**; any
  later Gemma-4 load failure is then known to be our compile, not the runtime.
- Fails at FastRPC/PD → apply the unsigned-PD contingency (step 4) and retry.
- Fails elsewhere → the forced-kDebug logs now show the real error; fix in this repo,
  rebuild .so, iterate — all via the CLI loop (minutes per cycle, no APK).

### Step 2 (optional) — same probe bundle through the app (few hours)
Only worth doing if step 1 ran. Temporary catalog entry in localai, rebuild
(platform-signed, `QNN_SDK_ROOT` = QAIRT 2.44), `install -r`, push bundle + `chcon`,
StatusActivity → NPU → "Test active model". Proves the app/JNI/binder path.

### Step 3 — **CRITICAL PATH: AOT-compile Gemma-4 E2B for V73** (Linux x86_64 box)
Follow `GEMMA4_V73_AOT_RUNBOOK.md` **after** verifying its flags (§2 issue 6):
`litert-torch export-hf --model=google/gemma-4-E2B-it --aot_backend=qualcomm
--aot_soc_model=SM8550` (SM8550 = V73; the AOT toolchain has no SM8735 entry — arch is
what matters, proven by SD running an SM8550-class binary on this device). Use QAIRT
2.42–2.44 to match the device runtime (System API 1.8). Start full-float to get
*loading* working, then optimize quantization. Expect **text-decoder-only** on NPU
(vision/audio encoder export is upstream-WIP) — keep vision/audio on GPU/CPU as the
localai config already does.

### Step 4 — contingency: unsigned-PD patch for arm64 (only if PD error appears)
The dispatch's unsigned-PD device config is `#if __x86_64__`-gated; arm64 gets the
default (signed) PD. Our app is platform-signed (may be allowed signed PD) — but if
step 1/2/3 hits `Untrusted application trying to offload to signed PD` or a FastRPC
failure at context-register, mirror localai `qnn_session.cpp`
(`QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD`, `useSignedProcessDomain=false`) in
`htp_backend.cc::Init` for the arm64 path, rebuild the .so.

### Step 5 — wire Gemma-4-V73 into localai + honest scope
Repoint `GEMMA4_E2B_NPU.fileName`, set `supportsVision/Audio=false` initially,
re-test multimodal separately. Update README + docs. Consider upstream PR (§3).

---

## 5. Reference

- **Device:** Nothing A024, SM8735 "sun", Hexagon **V73**, 12 GB. `adb root` works.
  Reports **fused socModel 91** via `QnnDevice_getPlatformInfo` (not 85!).
- **Version matrix (must all match):** litertlm-android **0.11.0** ↔ dispatch from
  LiteRT ~**v2.1.5** era ↔ **QAIRT 2.44.0.260225** (Qnn 2.33 / System API 1.8) ↔
  AOT compile with QAIRT 2.42–2.44. QAIRT 2.46 (Qnn 2.35) is too new; 2.45 too old.
- **localai branch:** `experiment/hexagon-npu-gemma` (pushed; remote moved to
  `lighteningAB/localai`). Contains toggle code, docs, the LiteRT patch file, and the
  rebuilt dispatch .so.
- **Staged on device** (`/data/local/tmp/`): `genie/` (QAIRT 2.44 runtime + V73 skel),
  `genie32/`, `genie3b/` (Llama bundles — version-locked, useless for litertlm, can be
  deleted to free ~5 GB). App `files/models/` has the (unusable, V79) sm8750 gemma
  bundle — delete to free 2.8 GB once step 1 passes.
- **Prior docs:** `GEMMA_NPU_DISPATCH_FIX.md` (diagnosis; §2/§3 partially superseded),
  `GEMMA4_V73_AOT_RUNBOOK.md` (AOT recipe), localai `README.md` "Hexagon NPU bundle",
  localai `docs/HEXAGON-NPU-GEMMA-PLAN.md` (stale copy).
