package com.nothing.localai.llm

import android.content.Context
import android.util.Log

private const val TAG = "EngineManager"

/**
 * Routes inference to the runtime that owns the active model's [Backend]:
 * LiteRT-LM ([LlmRunner]) or llama.cpp ([LlamaCppRunner]). The service and
 * SessionRegistry talk only to this — they never branch on backend.
 *
 * Only one runtime holds a model at a time (a 12B GGUF + a Gemma-4 .litertlm
 * together would blow the RAM budget), so a backend switch closes the other
 * engine first. The RAM floor ([ModelSpec.minRamBytes]) is enforced here: an
 * under-floor [setActiveModel] is refused and the previous model stays active,
 * so callers never trigger an OOM-killing load.
 */
class EngineManager(
    private val ctx: Context,
    downloader: LlmDownloader,
) : InferenceRunner {

    private val litert = LlmRunner(ctx, downloader)
    private val llama = LlamaCppRunner(ctx, downloader)

    private fun backendOf(id: String): Backend =
        ModelCatalog.byId(id)?.backend ?: Backend.LITERTLM

    private fun currentId(): String = ModelPrefs.getActiveModelId(ctx)

    private fun active(): InferenceRunner =
        if (backendOf(currentId()) == Backend.LLAMACPP) llama else litert

    override fun activeModelId(): String = currentId()

    override fun newSession(sessionId: String): InferenceSession =
        active().newSession(sessionId)

    override fun setActiveModel(newId: String): Boolean {
        val spec = ModelCatalog.byId(newId)
        if (spec == null) { Log.w(TAG, "setActiveModel: unknown $newId"); return false }
        if (newId == currentId()) return false
        if (!DeviceRam.meetsFloor(ctx, spec)) {
            Log.w(TAG, "setActiveModel: REFUSED $newId — device RAM ${DeviceRam.totalBytes(ctx)} " +
                "< floor ${spec.minRamBytes}; keeping ${currentId()}")
            return false
        }
        // Free BOTH engines before the swap so the outgoing model's RAM is
        // reclaimed before the incoming one loads (they can't coexist).
        runCatching { litert.close() }
        runCatching { llama.close() }
        ModelPrefs.setActiveModelId(ctx, newId)
        if (spec.backend == Backend.LLAMACPP) llama.adoptModel(newId) else litert.adoptModel(newId)
        Log.i(TAG, "active model -> $newId (${spec.backend})")
        return true
    }

    override fun cancel(requestId: String) {
        // requestIds are globally unique; signalling both runners is harmless.
        litert.cancel(requestId)
        llama.cancel(requestId)
    }

    override fun close() {
        runCatching { litert.close() }
        runCatching { llama.close() }
    }
}
