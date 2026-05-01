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
        SessionRegistry(runner, applicationContext, maxLive = 4)
    }

    private var activeRequests = 0

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        Log.d(TAG, "onBind ${intent.action}")
        return binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        Log.d(TAG, "onUnbind")
        sessions.releaseAll()
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
    }
}
