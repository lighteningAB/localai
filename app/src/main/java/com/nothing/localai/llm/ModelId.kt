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

    // Experiment branch: re-attempt Gemma 4 E4B on LiteRT-LM 0.11.0 (May 2026).
    // 0.10.2 SIGSEGV'd deterministically on the multimodal vision path on
    // Snapdragon 8s Gen 4. 0.11.0's changelog does not call out a vision fix,
    // but added Gemma 4 MTP (>2x GPU decode) — worth re-testing before
    // committing to either runtime for production. Flip back to
    // GEMMA3N_E4B_INT4 if vision crashes again.
    const val DEFAULT = GEMMA4_E4B_INT4
}

data class ModelStatus(
    val state: State,
    val bytesDownloaded: Long,
    val totalBytes: Long,
    val error: String? = null,
) {
    enum class State { MISSING, DOWNLOADING, READY, ERROR }
}

data class ModelSpec(
    val id: String,
    val fileName: String,
    val downloadUrl: String,
    val sha256: String? = null,
    val totalBytes: Long,
    val supportsVision: Boolean,
    val supportsAudio: Boolean,
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

    fun byId(id: String): ModelSpec? = when (id) {
        ModelId.GEMMA3N_E2B_INT4 -> GEMMA3N_E2B
        ModelId.GEMMA3N_E4B_INT4 -> GEMMA3N_E4B
        ModelId.GEMMA4_E2B_INT4 -> GEMMA4_E2B
        ModelId.GEMMA4_E4B_INT4 -> GEMMA4_E4B
        else -> null
    }
}
