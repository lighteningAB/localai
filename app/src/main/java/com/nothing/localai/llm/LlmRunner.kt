package com.nothing.localai.llm

import android.content.Context
import android.os.ParcelFileDescriptor
import android.os.RemoteException
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.nothing.localai.ITokenCallback
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "LlmRunner"
private const val MAX_IMAGES_PER_TURN = 4

/**
 * LiteRT-LM 0.11.0 runner — experiment branch chasing Gemma 4 E4B multimodal.
 *
 * Vision is enabled via `EngineConfig.visionBackend = Backend.GPU()`; LiteRT-LM
 * stages images through `Content.ImageFile(path)`, so [ChatSession.addImage]
 * materializes the incoming JPEG PFD to a temp file under cacheDir and deletes
 * after the generate Flow completes.
 *
 * Audio uses `EngineConfig.audioBackend = Backend.GPU()` (field name confirmed
 * by javap'ing litertlm-android-0.11.0; not in the public Android quickstart).
 * [ChatSession.addAudio] reads the incoming PCM16 PFD into memory and feeds it
 * as `Content.AudioBytes(byte[])` at generate time. Gemma 4's USM encoder
 * expects 16 kHz mono; a non-16k sampleRate is logged but not rejected.
 */
class LlmRunner(
    private val ctx: Context,
    private val downloader: LlmDownloader,
    private val modelId: String = ModelId.DEFAULT,
) {

    @Volatile private var engine: Engine? = null
    private val jobs = ConcurrentHashMap<String, Job>()
    val spec: ModelSpec get() = ModelCatalog.byId(modelId) ?: error("unknown model $modelId")

    @Synchronized
    fun engine(): Engine {
        engine?.let { return it }
        val modelFile: File = downloader.fileFor(spec)
        check(modelFile.exists()) { "model not present at ${modelFile.absolutePath}" }
        val cfg = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.CPU(),
            visionBackend = if (spec.supportsVision) Backend.GPU() else null,
            audioBackend = if (spec.supportsAudio) Backend.GPU() else null,
        )
        // initialize() can take ~10s for a 3.66 GB .litertlm — caller must be
        // off the main thread. Service binder calls already arrive on a pool
        // thread, and Aiwidget's bridge guarantees Dispatchers.IO upstream.
        return Engine(cfg).also {
            it.initialize()
            engine = it
        }
    }

    fun newConversation(): Conversation =
        engine().createConversation(ConversationConfig())

    fun registerJob(requestId: String, job: Job) {
        jobs[requestId] = job
    }

    fun cancel(requestId: String) {
        jobs[requestId]?.cancel()
    }

    fun forgetJob(requestId: String) {
        jobs.remove(requestId)
    }

    fun close() {
        engine?.close()
        engine = null
    }
}

/**
 * One [Conversation] per widget instance. Owns chat history via the
 * conversation's KV cache, plus any pending image files attached since the
 * last [generate] call.
 */
class ChatSession(
    private val sessionId: String,
    private val runner: LlmRunner,
    private val ctx: Context,
) {
    private var conversation: Conversation = runner.newConversation()
    private val pendingImages = mutableListOf<File>()
    private var pendingAudio: ByteArray? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Synchronized
    fun reset() {
        runCatching { conversation.close() }
        conversation = runner.newConversation()
        clearPending()
    }

    @Synchronized
    fun close() {
        runCatching { conversation.close() }
        clearPending()
        scope.cancel()
    }

    @Synchronized
    fun addImage(jpegFd: ParcelFileDescriptor) {
        if (!runner.spec.supportsVision) {
            Log.w(TAG, "[$sessionId] addImage on non-vision model; ignored")
            jpegFd.close(); return
        }
        if (pendingImages.size >= MAX_IMAGES_PER_TURN) {
            Log.w(TAG, "[$sessionId] addImage: max $MAX_IMAGES_PER_TURN per turn; ignored")
            jpegFd.close(); return
        }
        try {
            val cacheDir = File(ctx.cacheDir, "session-$sessionId").apply { mkdirs() }
            val out = File(cacheDir, "img-${UUID.randomUUID()}.jpg")
            ParcelFileDescriptor.AutoCloseInputStream(jpegFd).use { input ->
                out.outputStream().use { input.copyTo(it) }
            }
            pendingImages += out
            Log.d(TAG, "[$sessionId] image staged at ${out.absolutePath} (${out.length()} bytes)")
        } catch (t: Throwable) {
            Log.e(TAG, "[$sessionId] addImage failed", t)
            runCatching { jpegFd.close() }
            throw t
        }
    }

    @Synchronized
    fun addAudio(pcmFd: ParcelFileDescriptor, sampleRate: Int) {
        if (!runner.spec.supportsAudio) {
            Log.w(TAG, "[$sessionId] addAudio on non-audio model; ignored")
            pcmFd.close(); return
        }
        if (sampleRate != 16_000) {
            Log.w(TAG, "[$sessionId] addAudio: expected 16 kHz mono PCM16, got ${sampleRate}Hz — model will likely garbage out")
        }
        try {
            val bytes = ParcelFileDescriptor.AutoCloseInputStream(pcmFd).use { it.readBytes() }
            pendingAudio = bytes
            Log.d(TAG, "[$sessionId] audio staged: ${bytes.size} bytes @ ${sampleRate}Hz PCM16")
        } catch (t: Throwable) {
            Log.e(TAG, "[$sessionId] addAudio failed", t)
            runCatching { pcmFd.close() }
            throw t
        }
    }

    fun generate(requestId: String, prompt: String, cb: ITokenCallback) {
        val full = StringBuilder()
        val parts: List<Content>
        val toCleanup: List<File>
        synchronized(this) {
            val ps = mutableListOf<Content>()
            pendingImages.forEach { ps += Content.ImageFile(it.absolutePath) }
            pendingAudio?.let { ps += Content.AudioBytes(it) }
            if (prompt.isNotEmpty()) ps += Content.Text(prompt)
            parts = ps
            toCleanup = pendingImages.toList()
            pendingImages.clear()
            pendingAudio = null
        }

        val job = scope.launch {
            var errored = false
            try {
                conversation.sendMessageAsync(Contents.of(*parts.toTypedArray()))
                    .catch { t ->
                        errored = true
                        Log.e(TAG, "[$sessionId] generate flow failed", t)
                        safe { cb.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
                    }
                    .onCompletion {
                        toCleanup.forEach { runCatching { it.delete() } }
                        runner.forgetJob(requestId)
                    }
                    .collect { msg ->
                        val text = msg.toString()
                        if (text.isNotEmpty()) {
                            full.append(text)
                            safe { cb.onToken(requestId, text) }
                        }
                    }
                if (!errored) safe { cb.onDone(requestId, full.toString()) }
            } catch (ce: CancellationException) {
                Log.i(TAG, "[$sessionId] generate cancelled for $requestId")
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "[$sessionId] generate failed", t)
                safe { cb.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
            }
        }
        runner.registerJob(requestId, job)
    }

    private fun clearPending() {
        pendingImages.forEach { runCatching { it.delete() } }
        pendingImages.clear()
        pendingAudio = null
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (_: RemoteException) {}
    }
}
