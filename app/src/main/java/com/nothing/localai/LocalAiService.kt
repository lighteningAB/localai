package com.nothing.localai

import android.app.Notification
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.nothing.localai.imagegen.ImageGenRunner
import com.nothing.localai.imagegen.OutfitSwapRunner
import com.nothing.localai.llm.LlmDownloader
import com.nothing.localai.llm.LlmRunner
import com.nothing.localai.session.SessionRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

private const val TAG = "LocalAiService"
private const val API_VERSION = 1
private const val FG_NOTIFICATION_ID = 1001

class LocalAiService : LifecycleService() {

    private val downloader by lazy { LlmDownloader(applicationContext) }
    private val runner by lazy { LlmRunner(applicationContext, downloader) }
    private val sessions by lazy {
        // LiteRT-LM 0.11.0 caps the engine at one live Conversation; the
        // registry enforces single-active anyway, but pass 1 explicitly so
        // the parameter doesn't lie.
        SessionRegistry(runner, applicationContext, maxLive = 1)
    }
    // Diffusion runner is independent of the LLM engine. Held lazily so the
    // ImageGenerator (~seconds to construct, ~1–2 GB resident once loaded)
    // doesn't materialize until a widget actually asks for an image.
    private val imageGen by lazy { ImageGenRunner(applicationContext) }
    // Outfit-swap runner shares the diffusion bundle (CLIP + VAE) and adds the
    // SegFormer + SD 1.5 inpaint UNet binaries on top. See PLAN-OUTFIT-SWAP.md.
    private val outfitSwap by lazy { OutfitSwapRunner(applicationContext) }

    private var activeRequests = 0

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        Log.d(TAG, "onBind ${intent.action}")
        return binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        Log.d(TAG, "onUnbind")
        sessions.releaseAll()
        runCatching { imageGen.close() }
        runCatching { outfitSwap.close() }
        return super.onUnbind(intent)
    }

    private fun beginRequest() {
        if (activeRequests++ == 0) startForegroundCompat()
    }

    private fun endRequest() {
        if (--activeRequests <= 0) {
            activeRequests = 0
            stopForeground(STOP_FOREGROUND_REMOVE)
        }
    }

    private fun startForegroundCompat() {
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, com.nothing.localai.ui.StatusActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        val notif: Notification = NotificationCompat.Builder(this, LocalAiApp.CHANNEL_ID)
            .setContentTitle("Local AI running")
            .setContentText("On-device inference active")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(FG_NOTIFICATION_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(FG_NOTIFICATION_ID, notif)
        }
    }

    private val binder = object : ILocalAiService.Stub() {

        override fun getApiVersion(): Int = API_VERSION

        override fun getModelStatus(modelId: String): String {
            val s = downloader.statusOf(modelId)
            return JSONObject().apply {
                put("state", s.state.name.lowercase())
                put("bytesDownloaded", s.bytesDownloaded)
                put("totalBytes", s.totalBytes)
                put("error", s.error.orEmpty())
            }.toString()
        }

        override fun ensureModel(modelId: String, cb: IModelStatusCallback) {
            lifecycleScope.launch(Dispatchers.IO) {
                downloader.ensure(modelId, cb)
            }
        }

        override fun createSession(sessionId: String) {
            sessions.getOrCreate(sessionId)
        }

        override fun releaseSession(sessionId: String) {
            sessions.release(sessionId)
        }

        override fun resetSession(sessionId: String) {
            sessions.reset(sessionId)
        }

        override fun generate(sessionId: String, prompt: String, cb: ITokenCallback): String {
            val requestId = java.util.UUID.randomUUID().toString()
            beginRequest()
            // Wrap cb so endRequest() fires when generation completes — ChatSession
            // launches its own coroutine, so we can't bracket on this binder call.
            val wrapped = object : ITokenCallback.Stub() {
                override fun onToken(rid: String, text: String) = cb.onToken(rid, text)
                override fun onDone(rid: String, full: String) {
                    try { cb.onDone(rid, full) } finally { endRequest() }
                }
                override fun onError(rid: String, code: String, msg: String) {
                    try { cb.onError(rid, code, msg) } finally { endRequest() }
                }
            }
            try {
                sessions.getOrCreate(sessionId).generate(requestId, prompt, wrapped)
            } catch (t: Throwable) {
                Log.e(TAG, "generate failed", t)
                runCatching { wrapped.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
            }
            return requestId
        }

        override fun cancel(requestId: String) {
            sessions.cancel(requestId)
        }

        override fun addImage(sessionId: String, jpegFd: ParcelFileDescriptor) {
            try {
                sessions.getOrCreate(sessionId).addImage(jpegFd)
            } catch (t: Throwable) {
                Log.e(TAG, "addImage failed", t)
                runCatching { jpegFd.close() }
            }
        }

        override fun addAudio(sessionId: String, pcmFd: ParcelFileDescriptor, sampleRate: Int) {
            try {
                sessions.getOrCreate(sessionId).addAudio(pcmFd, sampleRate)
            } catch (t: Throwable) {
                Log.e(TAG, "addAudio failed", t)
                runCatching { pcmFd.close() }
            }
        }

        // ----- standalone vision/asr/tts (not used when chatting with Gemma 3n) -----

        override fun classifyImage(jpegFd: ParcelFileDescriptor, topK: Int): String {
            jpegFd.close()
            return JSONArray().toString()
        }

        override fun transcribe(pcmFd: ParcelFileDescriptor, sampleRate: Int): String {
            pcmFd.close()
            return ""
        }

        override fun speak(text: String) {
            // wired in later TTS phase
        }

        // ===== Image generation =====

        override fun generateImage(
            prompt: String,
            iterations: Int,
            seed: Long,
            cb: IImageGenCallback,
        ): String {
            beginRequest()
            // Wrap cb so endRequest() fires when generation completes. ImageGenRunner
            // launches its own coroutine, so we can't bracket on this binder call.
            val wrapped = object : IImageGenCallback.Stub() {
                override fun onStep(rid: String, step: Int, totalSteps: Int) =
                    cb.onStep(rid, step, totalSteps)
                override fun onStage(rid: String, stageName: String) {
                    // generateImage path doesn't emit stages, but the AIDL surface
                    // requires the method to be implemented. Relay defensively.
                    runCatching { cb.onStage(rid, stageName) }
                }
                override fun onResult(
                    rid: String,
                    pngFd: ParcelFileDescriptor,
                    width: Int,
                    height: Int,
                ) {
                    try { cb.onResult(rid, pngFd, width, height) } finally { endRequest() }
                }
                override fun onError(rid: String, code: String, msg: String) {
                    try { cb.onError(rid, code, msg) } finally { endRequest() }
                }
            }
            return try {
                imageGen.generate(prompt, iterations, seed, wrapped)
            } catch (t: Throwable) {
                Log.e(TAG, "generateImage failed", t)
                val rid = java.util.UUID.randomUUID().toString()
                runCatching { wrapped.onError(rid, "GENERATE_FAILED", t.message ?: "unknown") }
                rid
            }
        }

        override fun cancelImageGen(requestId: String) {
            imageGen.cancel(requestId)
        }

        // ===== Outfit swap =====

        override fun generateOutfitSwap(
            inputPng: ParcelFileDescriptor,
            prompt: String,
            garmentSpec: String,
            iterations: Int,
            seed: Long,
            cb: IImageGenCallback,
        ): String {
            beginRequest()
            val wrapped = object : IImageGenCallback.Stub() {
                override fun onStep(rid: String, step: Int, totalSteps: Int) =
                    cb.onStep(rid, step, totalSteps)
                override fun onStage(rid: String, stageName: String) =
                    cb.onStage(rid, stageName)
                override fun onResult(
                    rid: String,
                    pngFd: ParcelFileDescriptor,
                    width: Int,
                    height: Int,
                ) {
                    try { cb.onResult(rid, pngFd, width, height) } finally { endRequest() }
                }
                override fun onError(rid: String, code: String, msg: String) {
                    try { cb.onError(rid, code, msg) } finally { endRequest() }
                }
            }
            return try {
                outfitSwap.generate(inputPng, prompt, garmentSpec, iterations, seed, wrapped)
            } catch (t: Throwable) {
                Log.e(TAG, "generateOutfitSwap failed", t)
                val rid = java.util.UUID.randomUUID().toString()
                runCatching { wrapped.onError(rid, "GENERATE_FAILED", t.message ?: "unknown") }
                rid
            }
        }

        override fun cancelOutfitSwap(requestId: String) {
            outfitSwap.cancel(requestId)
        }
    }
}
