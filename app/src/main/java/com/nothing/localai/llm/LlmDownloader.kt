package com.nothing.localai.llm

import android.content.Context
import android.os.RemoteException
import android.util.Log
import com.nothing.localai.IModelStatusCallback
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.security.MessageDigest

private const val TAG = "LlmDownloader"

class LlmDownloader(private val ctx: Context) {

    private val http = OkHttpClient.Builder().build()

    private val modelsDir: File get() = File(ctx.filesDir, "models")

    /**
     * Returns the on-disk path for a model. Tries the spec's exact filename first,
     * then falls back to a case-insensitive scan of the models dir — this saves
     * us when a vendor renames a release artifact (E2B vs E2b, etc).
     */
    fun fileFor(spec: ModelSpec): File {
        val exact = File(modelsDir, spec.fileName)
        if (exact.exists()) return exact
        val match = modelsDir.listFiles()?.firstOrNull {
            it.name.equals(spec.fileName, ignoreCase = true)
        }
        return match ?: exact
    }

    fun statusOf(modelId: String): ModelStatus {
        val spec = ModelCatalog.byId(modelId)
            ?: return ModelStatus(ModelStatus.State.ERROR, 0, 0, "unknown model")
        val f = fileFor(spec)
        if (!f.exists()) return ModelStatus(ModelStatus.State.MISSING, 0, spec.totalBytes)
        return ModelStatus(ModelStatus.State.READY, f.length(), spec.totalBytes)
    }

    /**
     * Idempotent. If the file is already present and matches checksum (when set),
     * fires onReady immediately. Otherwise downloads atomically into filesDir.
     */
    fun ensure(modelId: String, cb: IModelStatusCallback) {
        val spec = ModelCatalog.byId(modelId) ?: run {
            safe { cb.onError(modelId, "UNKNOWN_MODEL", "no such model") }; return
        }
        val target = fileFor(spec)
        target.parentFile?.mkdirs()

        if (target.exists() && (spec.sha256 == null || sha256(target) == spec.sha256)) {
            safe { cb.onReady(modelId) }; return
        }
        if (spec.downloadUrl.isBlank()) {
            safe {
                cb.onError(
                    modelId, "NO_DOWNLOAD_URL",
                    "Model not bundled. Push manually: " +
                        "adb push <file> /data/local/tmp/${spec.fileName} && " +
                        "adb shell run-as com.nothing.localai.debug cp /data/local/tmp/${spec.fileName} files/models/"
                )
            }
            return
        }

        val tmp = File(target.parentFile, "${spec.fileName}.part")
        try {
            http.newCall(Request.Builder().url(spec.downloadUrl).build()).execute().use { resp ->
                if (!resp.isSuccessful) {
                    safe { cb.onError(modelId, "HTTP_${resp.code}", resp.message) }
                    return
                }
                val total = resp.body?.contentLength()?.takeIf { it > 0 } ?: spec.totalBytes
                tmp.outputStream().use { out ->
                    resp.body!!.byteStream().use { input ->
                        val buf = ByteArray(64 * 1024)
                        var got = 0L
                        while (true) {
                            val n = input.read(buf)
                            if (n <= 0) break
                            out.write(buf, 0, n)
                            got += n
                            safe { cb.onProgress(modelId, got, total) }
                        }
                    }
                }
            }
            if (spec.sha256 != null && sha256(tmp) != spec.sha256) {
                tmp.delete()
                safe { cb.onError(modelId, "CHECKSUM_MISMATCH", "sha256 did not match") }
                return
            }
            if (!tmp.renameTo(target)) {
                safe { cb.onError(modelId, "RENAME_FAILED", "could not move into place") }
                return
            }
            safe { cb.onReady(modelId) }
        } catch (t: Throwable) {
            Log.e(TAG, "download failed", t)
            tmp.delete()
            safe { cb.onError(modelId, "IO_ERROR", t.message ?: "unknown") }
        }
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (_: RemoteException) { /* client died */ }
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { ins ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = ins.read(buf); if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
