package com.nothing.localai.llm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.ParcelFileDescriptor
import android.os.RemoteException
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.nothing.localai.ITokenCallback
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "LlmRunner"
private const val MAX_IMAGES_PER_TURN = 4

/**
 * MediaPipe LlmInference runner. Reverted from LiteRT-LM (Engine/Conversation)
 * because LiteRT-LM 0.10.2 has a deterministic SIGSEGV on the multimodal
 * vision path for Gemma 4 — Google's own samples still target Gemma 3 here.
 *
 * Vision modality is enabled. Audio modality stays disabled in session config
 * because setEnableAudioModality(true) without AudioModelOptions causes the
 * inference graph to fail at open with "audio options should not be null".
 * That's TODO — wire AudioModelOptions when we revisit the voice widget.
 */
class LlmRunner(
    private val ctx: Context,
    private val downloader: LlmDownloader,
    private val modelId: String = ModelId.DEFAULT,
) {

    @Volatile private var engine: LlmInference? = null
    private val cancelFlags = ConcurrentHashMap<String, AtomicBoolean>()
    val spec: ModelSpec get() = ModelCatalog.byId(modelId) ?: error("unknown model $modelId")

    @Synchronized
    fun engine(): LlmInference {
        engine?.let { return it }
        val modelFile: File = downloader.fileFor(spec)
        check(modelFile.exists()) { "model not present at ${modelFile.absolutePath}" }
        // NOTE on audio: setAudioModelOptions(...) is intentionally NOT called.
        // It would cause engine init to fail with "tf_lite_audio_encoder_hw not
        // found" — the Gemma 3n preview .task ships only TF_LITE_PREFILL_DECODE,
        // EMBEDDER, PER_LAYER_EMBEDDER, VISION_ENCODER, VISION_ADAPTER. No audio
        // encoder weights. Re-add this call (and re-enable audio modality on
        // the session) once Google ships a .task with audio encoder included.
        val opts = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)
            .setMaxTokens(2048)
            .setMaxNumImages(if (spec.supportsVision) MAX_IMAGES_PER_TURN else 0)
            .build()
        return LlmInference.createFromOptions(ctx, opts).also { engine = it }
    }

    fun newSession(): LlmInferenceSession {
        val graph = GraphOptions.builder()
            .setEnableVisionModality(spec.supportsVision)
            // Audio modality is force-disabled even when spec.supportsAudio==true.
            // The Gemma 3n preview .task does NOT include audio encoder weights
            // (Google README: "current checkpoint only supports text and vision
            // input. We are actively working to roll out full multimodal..."),
            // so toggling audio causes engine init to fail with
            // "tf_lite_audio_encoder_hw not found". Re-enable when Google ships
            // a .task with audio encoder included.
            .setEnableAudioModality(false)
            .build()
        val opts = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(40)
            .setTemperature(0.7f)
            .setGraphOptions(graph)
            .build()
        return LlmInferenceSession.createFromOptions(engine(), opts)
    }

    fun registerCancel(requestId: String): AtomicBoolean {
        val flag = AtomicBoolean(false)
        cancelFlags[requestId] = flag
        return flag
    }

    fun cancel(requestId: String) {
        cancelFlags[requestId]?.set(true)
    }

    fun forgetCancel(requestId: String) {
        cancelFlags.remove(requestId)
    }

    fun close() {
        engine?.close()
        engine = null
    }
}

/**
 * One MediaPipe [LlmInferenceSession] per widget instance. Owns chat history
 * via the session's KV cache, and any pending multimodal chunks attached
 * since the last generate() call.
 */
class ChatSession(
    private val sessionId: String,
    private val runner: LlmRunner,
    private val ctx: Context,
) {
    private var session: LlmInferenceSession = runner.newSession()
    private var pendingImage: Bitmap? = null
    private var pendingAudio: ByteArray? = null

    @Synchronized
    fun reset() {
        runCatching { session.close() }
        session = runner.newSession()
        pendingImage?.recycle()
        pendingImage = null
        pendingAudio = null
    }

    @Synchronized
    fun close() {
        runCatching { session.close() }
        pendingImage?.recycle()
        pendingImage = null
        pendingAudio = null
    }

    @Synchronized
    fun addImage(jpegFd: ParcelFileDescriptor) {
        if (!runner.spec.supportsVision) {
            Log.w(TAG, "[$sessionId] addImage on non-vision model; ignored")
            jpegFd.close(); return
        }
        // Decode and store as a Bitmap; the next generate() call will feed it
        // into the session via session.addImage(MPImage). Centralizing the
        // attach happens at generate-time so order vs addQueryChunk is right.
        try {
            val bmp: Bitmap = ParcelFileDescriptor.AutoCloseInputStream(jpegFd).use { input ->
                val opts = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }
                BitmapFactory.decodeStream(input, null, opts)
                    ?: error("could not decode JPEG")
            }
            pendingImage?.recycle()
            pendingImage = bmp
            Log.d(TAG, "[$sessionId] image attached: ${bmp.width}x${bmp.height}")
        } catch (t: Throwable) {
            Log.e(TAG, "[$sessionId] addImage failed", t)
            runCatching { jpegFd.close() }
            throw t
        }
    }

    @Synchronized
    fun addAudio(pcmFd: ParcelFileDescriptor, sampleRate: Int) {
        // Drop audio: the Gemma 3n preview .task lacks audio encoder weights
        // (see comment in LlmRunner.newSession). Calling session.addAudio with
        // audio modality disabled would error; refuse here with a clear log.
        Log.w(TAG, "[$sessionId] addAudio: audio modality unavailable in current Gemma 3n checkpoint; dropping ${sampleRate}Hz PCM")
        runCatching { pcmFd.close() }
        throw UnsupportedOperationException(
            "Audio not supported by the current Gemma 3n preview checkpoint. " +
                "Voice transcription will be re-enabled once Google ships a .task with audio encoder weights."
        )
    }

    fun generate(requestId: String, prompt: String, cb: ITokenCallback) {
        val cancel = runner.registerCancel(requestId)
        val full = StringBuilder()
        try {
            synchronized(this) {
                pendingImage?.let { bmp ->
                    val mpImage = BitmapImageBuilder(bmp).build()
                    session.addImage(mpImage)
                    bmp.recycle()
                    pendingImage = null
                }
                pendingAudio?.let { bytes ->
                    session.addAudio(bytes)
                    pendingAudio = null
                }
                if (prompt.isNotEmpty()) session.addQueryChunk(prompt)
            }
            session.generateResponseAsync { partial, done ->
                if (cancel.get()) return@generateResponseAsync
                if (partial.isNotEmpty()) {
                    full.append(partial)
                    safe { cb.onToken(requestId, partial) }
                }
                if (done) {
                    runner.forgetCancel(requestId)
                    safe { cb.onDone(requestId, full.toString()) }
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "[$sessionId] generate failed", t)
            runner.forgetCancel(requestId)
            safe { cb.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
        }
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (_: RemoteException) {}
    }
}
