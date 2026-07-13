package com.nothing.localai.llm

import android.content.Context
import android.os.ParcelFileDescriptor
import android.os.RemoteException
import android.util.Log
import com.nothing.localai.ITokenCallback
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "LlamaCppRunner"
private const val CTX_TOKENS = 4096
private const val MAX_GEN_TOKENS = 1024

/**
 * llama.cpp runner (libllmcpp.so via [NativeLlama]) — the [Backend.LLAMACPP]
 * counterpart to [LlmRunner]. Loads a GGUF once (weights ~ the "engine") and
 * hands out [LlamaSession]s (one KV cache each). CPU-only: this is precisely
 * the path that lets Gemma 4 12B run despite the LiteRT-LM GPU 1 GB alloc cap.
 *
 * Single loaded model at a time, mirroring [LlmRunner]; [EngineManager] tears
 * down the LiteRT engine before this one comes up (and vice-versa).
 */
class LlamaCppRunner(
    private val ctx: Context,
    private val downloader: LlmDownloader,
) : InferenceRunner {

    @Volatile private var modelPtr: Long = 0L
    @Volatile private var modelId: String = ModelPrefs.getActiveModelId(ctx)

    private val spec: ModelSpec get() = ModelCatalog.byId(modelId) ?: error("unknown model $modelId")

    // requestId -> (owning session, coroutine job). The session (not a captured
    // ctxPtr) resolves the live context lazily, so cancel works even if the KV
    // context is still being created when cancel arrives.
    private data class Running(val session: LlamaSession, val job: Job)
    private val jobs = ConcurrentHashMap<String, Running>()

    override fun activeModelId(): String = modelId

    /** Load (or return) the GGUF weights. Throws with a clear message if the
     *  native lib is missing, the device is under the model's RAM floor, or the
     *  file isn't present. Off-main; a 7 GB mmap is ~seconds. */
    @Synchronized
    fun model(): Long {
        if (modelPtr != 0L) return modelPtr
        check(NativeLlama.ensureLoaded()) { "libllmcpp.so not available on this device" }
        val s = spec
        check(DeviceRam.meetsFloor(ctx, s)) {
            "insufficient RAM for ${s.id}: needs ${s.minRamBytes} bytes, device has ${DeviceRam.totalBytes(ctx)}"
        }
        val f: File = downloader.fileFor(s)
        check(f.exists()) { "model not present at ${f.absolutePath}" }
        Log.i(TAG, "loading GGUF: ${f.absolutePath}")
        val p = NativeLlama.nativeLoadModel(f.absolutePath)
        check(p != 0L) { "llama_model_load_from_file failed for ${f.absolutePath}" }
        modelPtr = p
        return p
    }

    override fun newSession(sessionId: String): InferenceSession =
        LlamaSession(sessionId, this, ctx)

    @Synchronized
    override fun setActiveModel(newId: String): Boolean {
        if (ModelCatalog.byId(newId) == null) {
            Log.w(TAG, "setActiveModel: unknown model $newId"); return false
        }
        if (newId == modelId) return false
        Log.i(TAG, "switching model $modelId -> $newId")
        ModelPrefs.setActiveModelId(ctx, newId)
        freeModel()
        modelId = newId
        return true
    }

    fun createContext(): Long {
        val m = model()
        val nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8)
        val ctxPtr = NativeLlama.nativeCreateContext(m, CTX_TOKENS, nThreads, /*seed=*/42)
        check(ctxPtr != 0L) { "llama context creation failed" }
        return ctxPtr
    }

    fun modelPtrOrLoad(): Long = model()

    fun registerJob(requestId: String, session: LlamaSession, job: Job) {
        jobs[requestId] = Running(session, job)
    }

    fun forgetJob(requestId: String) {
        jobs.remove(requestId)
    }

    override fun cancel(requestId: String) {
        jobs[requestId]?.let {
            it.session.signalCancel()
            it.job.cancel()
        }
    }

    /** Adopt an active model id chosen by [EngineManager]. Frees the loaded
     *  GGUF so the next [model] call reloads against [newId]. */
    @Synchronized
    fun adoptModel(newId: String) {
        freeModel()
        modelId = newId
    }

    @Synchronized
    private fun freeModel() {
        if (modelPtr != 0L) {
            runCatching { NativeLlama.nativeFreeModel(modelPtr) }
            modelPtr = 0L
        }
    }

    @Synchronized
    override fun close() = freeModel()
}

/**
 * One llama.cpp chat context (KV cache). Text-only: [addImage]/[addAudio] are
 * ignored because the 12B GGUF ships without the vision projector.
 */
class LlamaSession(
    private val sessionId: String,
    private val runner: LlamaCppRunner,
    @Suppress("unused") private val ctx: Context,
) : InferenceSession {

    @Volatile private var ctxPtr: Long = 0L
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Synchronized
    private fun ensureCtx(): Long {
        if (ctxPtr == 0L) ctxPtr = runner.createContext()
        return ctxPtr
    }

    override fun generate(requestId: String, prompt: String, cb: ITokenCallback) =
        run(requestId, prompt, grammar = "", cb = cb)

    override fun generateConstrained(
        requestId: String, prompt: String, grammar: String, cb: ITokenCallback,
    ) = run(requestId, prompt, grammar, cb)

    private fun run(requestId: String, prompt: String, grammar: String, cb: ITokenCallback) {
        val job = scope.launch {
            val full = StringBuilder()
            try {
                val c = ensureCtx()
                val model = runner.modelPtrOrLoad()
                val sink = object : NativeLlama.TokenSink {
                    override fun onToken(text: String) {
                        full.append(text)
                        safe { cb.onToken(requestId, text) }
                    }
                }
                // Blocking, streams via sink on this IO thread.
                val out = NativeLlama.nativeGenerate(c, model, prompt, grammar, MAX_GEN_TOKENS, sink)
                val result = if (out.isNotEmpty()) out else full.toString()
                safe { cb.onDone(requestId, result) }
            } catch (ce: CancellationException) {
                Log.i(TAG, "[$sessionId] generate cancelled for $requestId")
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "[$sessionId] generate failed", t)
                safe { cb.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
            } finally {
                runner.forgetJob(requestId)
            }
        }
        runner.registerJob(requestId, this, job)
    }

    /** Signal the native decode loop to stop between tokens (see [close]). */
    fun signalCancel() {
        if (ctxPtr != 0L) runCatching { NativeLlama.nativeCancel(ctxPtr) }
    }

    override fun addImage(jpegFd: ParcelFileDescriptor) {
        Log.w(TAG, "[$sessionId] addImage ignored (llama.cpp text-only model)")
        runCatching { jpegFd.close() }
    }

    override fun addAudio(pcmFd: ParcelFileDescriptor, sampleRate: Int) {
        Log.w(TAG, "[$sessionId] addAudio ignored (llama.cpp text-only model)")
        runCatching { pcmFd.close() }
    }

    @Synchronized
    override fun reset() {
        if (ctxPtr != 0L) runCatching { NativeLlama.nativeResetContext(ctxPtr) }
    }

    @Synchronized
    override fun close() {
        if (ctxPtr != 0L) {
            runCatching { NativeLlama.nativeFreeContext(ctxPtr) }
            ctxPtr = 0L
        }
        scope.cancel()
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (_: RemoteException) {}
    }
}
