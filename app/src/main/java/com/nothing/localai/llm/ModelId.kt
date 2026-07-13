package com.nothing.localai.llm

object ModelId {
    // Gemma 3n — text + image + audio multimodal. Known-good MediaPipe .task.
    const val GEMMA3N_E2B_INT4 = "gemma3n-e2b-it-int4"
    const val GEMMA3N_E4B_INT4 = "gemma3n-e4b-it-int4"

    // Gemma 4 — successor to 3n. text + image + audio + video-via-frames.
    // 5.1B raw / 2.3B effective for E2B. The MediaPipe .task variant lives on
    // a gated HF repo; verify the exact filename after license acceptance.
    const val GEMMA4_E2B_INT4 = "gemma4-e2b-it-int4"
    const val GEMMA4_E4B_INT4 = "gemma4-e4b-it-int4"

    // Gemma 4 E2B, Qualcomm NPU-specialized .litertlm. Runs the decoder on the
    // Hexagon NPU via QNN/QAIRT instead of XNNPACK CPU. Selectable at runtime
    // from StatusActivity; see ModelPrefs / LlmRunner.setActiveModel.
    const val GEMMA4_E2B_NPU = "gemma4-e2b-it-npu"

    // Gemma 4 12B, QAT Q4_0 GGUF, run on the CPU via llama.cpp (libllmcpp.so).
    // The LiteRT-LM GPU runtime can't host the 12B (Adreno 1 GB logits-alloc
    // cap; LiteRT-LM #2461). RAM-gated to flagship devices — see ModelSpec.
    const val GEMMA4_12B_QAT = "gemma4-12b-it-qat"

    // Experiment branch: re-attempt Gemma 4 E4B on LiteRT-LM 0.11.0 (May 2026).
    // 0.10.2 SIGSEGV'd deterministically on the multimodal vision path on
    // Snapdragon 8s Gen 4. 0.11.0's changelog does not call out a vision fix,
    // but added Gemma 4 MTP (>2x GPU decode) — worth re-testing before
    // committing to either runtime for production. Flip back to
    // GEMMA3N_E4B_INT4 if vision crashes again.
    const val DEFAULT = GEMMA4_E2B_INT4
}

data class ModelStatus(
    val state: State,
    val bytesDownloaded: Long,
    val totalBytes: Long,
    val error: String? = null,
) {
    enum class State { MISSING, DOWNLOADING, READY, ERROR }
}

/**
 * Which compute backend [LlmRunner] hands to LiteRT-LM for this bundle.
 *
 * - [CPU_GPU]: stock generic .litertlm — text decoder on XNNPACK CPU, vision
 *   encoder on Adreno GPU (OpenCL), audio encoder on CPU. Works on any chip.
 * - [NPU]: Qualcomm NPU-specialized .litertlm — decoder runs on the Hexagon
 *   NPU via QNN/QAIRT (`Backend.NPU(nativeLibraryDir)`). The bundle must be
 *   the chip-matched artifact from the litert-community repo.
 */
enum class Accelerator { CPU_GPU, NPU }

/**
 * Which inference runtime hosts this model.
 *
 * - [LITERTLM]: the Kotlin `com.google.ai.edge.litertlm` path ([LlmRunner]).
 *   The default for every Gemma 4 .litertlm / .task bundle.
 * - [LLAMACPP]: the native llama.cpp engine ([LlamaCppRunner] / libllmcpp.so),
 *   for GGUF models the LiteRT-LM runtime can't host. Gemma 4 12B lives here:
 *   its fp32 logits tensor overflows the Adreno 1 GB single-allocation cap on
 *   the LiteRT-LM GPU backend (LiteRT-LM issue #2461), so it runs on the CPU.
 */
enum class Backend { LITERTLM, LLAMACPP }

data class ModelSpec(
    val id: String,
    val fileName: String,
    val downloadUrl: String,
    val sha256: String? = null,
    val totalBytes: Long,
    val supportsVision: Boolean,
    val supportsAudio: Boolean,
    val accelerator: Accelerator = Accelerator.CPU_GPU,
    val backend: Backend = Backend.LITERTLM,
    /**
     * Minimum device RAM (bytes, ActivityManager.MemoryInfo.totalMem) required
     * to even attempt this model. 0 = no floor. The 12B needs ~7 GB resident
     * weights + KV + activations; attempting it on an 8 GB phone OOM-kills the
     * process, so the service refuses below this and surfaces a clean error.
     */
    val minRamBytes: Long = 0,
)

object ModelCatalog {
    // Gemma 3n .task bundles are gated on HF/Kaggle by license acceptance.
    // Until you wire a hosting URL, push manually — see README.md.
    val GEMMA3N_E2B = ModelSpec(
        id = ModelId.GEMMA3N_E2B_INT4,
        fileName = "gemma-3n-E2B-it-int4.task",
        downloadUrl = "",
        sha256 = null,
        totalBytes = 3_140_000_000L, // ~2.9 GiB
        supportsVision = true,
        supportsAudio = true,
    )

    val GEMMA3N_E4B = ModelSpec(
        id = ModelId.GEMMA3N_E4B_INT4,
        fileName = "gemma-3n-E4B-it-int4.task",
        downloadUrl = "",
        sha256 = null,
        totalBytes = 4_410_000_000L,
        supportsVision = true,
        supportsAudio = true,
    )

    // Gemma 4 .litertlm bundles from litert-community/gemma-4-{E2B,E4B}-it-litert-lm.
    // The same repos also ship a *-web.task variant (browser-only, won't load on
    // Android) and Qualcomm NPU-specialized .litertlm files we don't use here.
    val GEMMA4_E2B = ModelSpec(
        id = ModelId.GEMMA4_E2B_INT4,
        fileName = "gemma-4-E2B-it.litertlm",
        downloadUrl = "",
        sha256 = null,
        totalBytes = 2_580_000_000L, // ~2.58 GB
        supportsVision = true,
        supportsAudio = true,
    )

    val GEMMA4_E4B = ModelSpec(
        id = ModelId.GEMMA4_E4B_INT4,
        fileName = "gemma-4-E4B-it.litertlm",
        downloadUrl = "",
        sha256 = null,
        totalBytes = 3_650_000_000L, // ~3.65 GB
        supportsVision = true,
        supportsAudio = true,
    )

    // Qualcomm NPU-specialized Gemma 4 E2B from litert-community. Per-chip
    // bundles exist (qualcomm_sm8750 / qualcomm_qcs8275); SM8735 has none, so we
    // use the sm8750 (Hexagon V79) artifact and rely on HTP forward-compat. Push:
    //   adb push gemma-4-E2B-it_qualcomm_sm8750.litertlm /data/local/tmp/
    //   adb shell run-as com.nothing.localai.debug \
    //     cp /data/local/tmp/gemma-4-E2B-it_qualcomm_sm8750.litertlm files/models/
    //
    // NOTE (verified on device 2026-06-30): selecting this currently fails at
    // `litert: No dispatch library found in .../lib/arm64`. The litertlm-android
    // 0.11.0 AAR does NOT ship the QNN dispatch plugin (libLiteRtDispatch_Qualcomm.so,
    // identified by exported symbol LiteRtDispatchGetApi). It must be built from
    // @litert//litert/vendors/qualcomm/dispatch:dispatch_api_so and staged into
    // jniLibs (ABI must match the AAR — see google-ai-edge/LiteRT issue #6889).
    // QNN runtime + V73 Skel are already packaged; libQnnSystem must be >= 1.8.0
    // (QAIRT 2.46 ok; 2.45 ships 1.7.0 and fails — LiteRT-LM issue #2226).
    val GEMMA4_E2B_NPU = ModelSpec(
        id = ModelId.GEMMA4_E2B_NPU,
        // Self-compiled Hexagon V73 build (AOT via litert-torch --aot_soc_model=SM8550,
        // which is V73 = same arch as SM8735; the published sm8750 bundle is V79 and
        // won't load). Text decoder only (dynamic_wi8_afp32); vision/audio off.
        fileName = "gemma-4-E2B-it_qualcomm_sm8735.litertlm",
        downloadUrl = "",
        sha256 = null,
        totalBytes = 5_103_785_936L, // ~4.8 GB (int8-weight V73 build)
        supportsVision = false,
        supportsAudio = false,
        accelerator = Accelerator.NPU,
    )

    // Gemma 4 12B QAT Q4_0 GGUF (google/gemma-4-12B-it-qat-q4_0-gguf). Runs on
    // the CPU via llama.cpp — sidesteps the Adreno 1 GB logits-alloc cap that
    // blocks the .litertlm 12B on the GPU (LiteRT-LM #2461). Text-only for now
    // (the repo's mmproj vision projector is not bundled). Push:
    //   adb push gemma-4-12b-it-qat-q4_0.gguf /data/local/tmp/
    //   adb shell run-as com.nothing.localai.debug \
    //     cp /data/local/tmp/gemma-4-12b-it-qat-q4_0.gguf files/models/
    val GEMMA4_12B_QAT = ModelSpec(
        id = ModelId.GEMMA4_12B_QAT,
        fileName = "gemma-4-12b-it-qat-q4_0.gguf",
        downloadUrl = "",
        sha256 = null,
        totalBytes = 6_975_877_728L, // ~6.5 GiB
        supportsVision = false,
        supportsAudio = false,
        backend = Backend.LLAMACPP,
        // ~7 GB weights + KV + activations. Require ~12 GB total RAM so we never
        // attempt it on 8 GB devices (guaranteed OOM). Canoe (24 GB) passes.
        minRamBytes = 12_000_000_000L,
    )

    fun byId(id: String): ModelSpec? = when (id) {
        ModelId.GEMMA3N_E2B_INT4 -> GEMMA3N_E2B
        ModelId.GEMMA3N_E4B_INT4 -> GEMMA3N_E4B
        ModelId.GEMMA4_E2B_INT4 -> GEMMA4_E2B
        ModelId.GEMMA4_E4B_INT4 -> GEMMA4_E4B
        ModelId.GEMMA4_E2B_NPU -> GEMMA4_E2B_NPU
        ModelId.GEMMA4_12B_QAT -> GEMMA4_12B_QAT
        else -> null
    }

    /** Models offered in the StatusActivity switch, in display order. */
    val selectable: List<ModelSpec> = listOf(GEMMA4_E4B, GEMMA4_E2B, GEMMA4_E2B_NPU, GEMMA4_12B_QAT)
}
