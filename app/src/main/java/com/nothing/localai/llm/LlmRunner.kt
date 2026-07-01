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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "LlmRunner"
private const val MAX_IMAGES_PER_TURN = 4

/**
 * Wrap raw PCM16 mono samples in a 44-byte WAV header. `Content.AudioBytes`
 * runs the bytes through miniaudio, which only recognizes encoded container
 * formats (WAV/MP3/FLAC/...). Without this wrapper miniaudio errors out with
 * "Failed to initialize miniaudio decoder, error code: -10" (MA_INVALID_FILE).
 */
private fun wrapPcm16MonoAsWav(pcm: ByteArray, sampleRate: Int): ByteArray {
    val byteRate = sampleRate * 2 // mono, 16-bit
    val dataSize = pcm.size
    val bb = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)
    bb.put("RIFF".toByteArray(Charsets.US_ASCII))
    bb.putInt(36 + dataSize)
    bb.put("WAVE".toByteArray(Charsets.US_ASCII))
    bb.put("fmt ".toByteArray(Charsets.US_ASCII))
    bb.putInt(16)              // fmt chunk size
    bb.putShort(1)             // PCM
    bb.putShort(1)             // mono
    bb.putInt(sampleRate)
    bb.putInt(byteRate)
    bb.putShort(2)             // block align: 1 ch * 16 bit / 8
    bb.putShort(16)            // bits per sample
    bb.put("data".toByteArray(Charsets.US_ASCII))
    bb.putInt(dataSize)
    bb.put(pcm)
    return bb.array()
}

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
) {

    @Volatile private var engine: Engine? = null
    // Active model is persisted in :inference's ModelPrefs and swappable at
    // runtime via [setActiveModel] (driven by StatusActivity over the binder).
    @Volatile private var modelId: String = ModelPrefs.getActiveModelId(ctx)
    private val jobs = ConcurrentHashMap<String, Job>()
    val spec: ModelSpec get() = ModelCatalog.byId(modelId) ?: error("unknown model $modelId")

    fun activeModelId(): String = modelId

    @Synchronized
    fun engine(): Engine {
        engine?.let { return it }
        val modelFile: File = downloader.fileFor(spec)
        check(modelFile.exists()) { "model not present at ${modelFile.absolutePath}" }
        // Text/decoder backend. CPU_GPU bundles run the decoder on XNNPACK CPU;
        // the Qualcomm NPU bundle runs it on the Hexagon NPU via QNN/QAIRT.
        // Backend.NPU needs a real filesystem path so FastRPC can load the
        // per-Hexagon-rev Skel — same nativeLibraryDir the imagegen QNN path uses.
        val textBackend = when (spec.accelerator) {
            // Run the Gemma-4 decoder on the Adreno GPU (LiteRT-LM 0.11.0 added
            // Gemma-4 MTP GPU decode, ~2x). If Engine.initialize() throws a text
            // backend constraint mismatch, fall back to Backend.CPU().
            Accelerator.CPU_GPU -> Backend.GPU()
            Accelerator.NPU -> {
                val nativeDir = ctx.applicationInfo.nativeLibraryDir
                // Backend.NPU takes the dir LiteRT scans for the dispatch plugin
                // (a .so exporting LiteRtDispatchGetApi — NOT shipped in the
                // litertlm 0.11.0 AAR; see README) AND for version-matched QNN
                // libs (litert prepends this dir to the QNN search path). The AAR's
                // libLiteRt expects a SPECIFIC QNN version (Qnn 2.33 / System 1.8 /
                // QAIRT 2.44.0.260225) — our app-bundled QNN is 2.46 (Qnn 2.35),
                // too new, which makes `dispatch_api.cc: Failed to set up QNN manager`.
                // So we stage the matched dispatch + QNN libs (Htp/System/Stub/Skel)
                // in files/models and point both LiteRT and FastRPC there, isolated
                // from the imagegen SD path which keeps using 2.46 in nativeLibraryDir.
                // nativeLibraryDir is read-only post-install, hence files/models.
                val modelsDir = downloader.fileFor(spec).parentFile
                val dispatchDir = modelsDir?.takeIf { d ->
                    d.listFiles()?.any {
                        it.isFile && it.name.contains("ispatch", ignoreCase = true) &&
                            it.name.endsWith(".so")
                    } == true
                }?.absolutePath ?: nativeDir
                // FastRPC loads the per-Hexagon-rev Skel by filesystem path via
                // ADSP_LIBRARY_PATH; point it at the dispatch dir first so the
                // matched V73 Skel staged there wins over the 2.46 one in
                // nativeLibraryDir, then fall back to nativeDir. Set before
                // initialize() to beat LocalAiApp's daemon-probe race.
                runCatching {
                    com.nothing.localai.imagegen.NativeImageGen
                        .nativeSetAdspLibraryPath("$dispatchDir:$nativeDir")
                }.onFailure { Log.w(TAG, "nativeSetAdspLibraryPath failed (NPU)", it) }
                Log.i(TAG, "NPU dispatch dir = $dispatchDir (adsp=$dispatchDir:$nativeDir)")
                Backend.NPU(dispatchDir)
            }
        }
        val cfg = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = textBackend,
            // Vision encoder stays on the Adreno GPU even for the NPU bundle —
            // the Qualcomm artifact specializes the decoder, not the SigLIP
            // tower. If Engine.initialize() throws a vision backend constraint
            // mismatch, flip this to Backend.NPU(...) / Backend.CPU() per the
            // error's "Model requires one of [...]" hint.
            visionBackend = if (spec.supportsVision) Backend.GPU() else null,
            // gemma-4-E4B-it.litertlm's audio encoder is CPU-only — Engine.initialize()
            // throws "Audio backend constraint mismatch. Model requires one of [cpu]
            // but Audio backend is GPU" if this is GPU. Vision is GPU-capable; audio
            // isn't (yet). Keep CPU until a future bundle lifts the constraint.
            audioBackend = if (spec.supportsAudio) Backend.CPU() else null,
        )
        Log.i(TAG, "initializing engine: model=$modelId accel=${spec.accelerator} path=${modelFile.absolutePath}")
        // initialize() can take ~10s for a 3.66 GB .litertlm — caller must be
        // off the main thread. Service binder calls already arrive on a pool
        // thread, and Aiwidget's bridge guarantees Dispatchers.IO upstream.
        return Engine(cfg).also {
            it.initialize()
            engine = it
        }
    }

    /**
     * Swap the active model. Persists the choice and tears down the live engine
     * so the next [engine] call rebuilds against the new bundle. The caller MUST
     * close every open [Conversation] first (SessionRegistry.releaseAll) — the
     * engine refuses to close while a Conversation is open against it.
     *
     * No-op (returns false) if [modelId] is already active or unknown.
     */
    @Synchronized
    fun setActiveModel(newId: String): Boolean {
        if (ModelCatalog.byId(newId) == null) {
            Log.w(TAG, "setActiveModel: unknown model $newId")
            return false
        }
        if (newId == modelId) return false
        Log.i(TAG, "switching model $modelId -> $newId")
        ModelPrefs.setActiveModelId(ctx, newId)
        engine?.let { runCatching { it.close() } }
        engine = null
        modelId = newId
        return true
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
            val pcm = ParcelFileDescriptor.AutoCloseInputStream(pcmFd).use { it.readBytes() }
            val wav = wrapPcm16MonoAsWav(pcm, sampleRate)
            pendingAudio = wav
            Log.d(TAG, "[$sessionId] audio staged: ${pcm.size} bytes PCM16 -> ${wav.size} bytes WAV @ ${sampleRate}Hz")
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
