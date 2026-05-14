package com.nothing.localai

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.nothing.localai.imagegen.DEFAULT_DIFFUSION_DIR_NAME
import com.nothing.localai.imagegen.NativeImageGen
import java.io.File
import kotlin.concurrent.thread

class LocalAiApp : Application() {
    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Local AI",
                NotificationManager.IMPORTANCE_MIN
            )
        )
        // Run the imagegen probe on a background thread — it includes a
        // full UNet diffusion pass when the bundle is present, which would
        // ANR the main thread in onCreate().
        thread(start = true, isDaemon = true, name = "imagegen-probe") {
            try {
                probeImagegenNative()
            } catch (t: Throwable) {
                Log.w(TAG, "imagegen probe thread crashed", t)
            }
        }
    }

    private fun probeImagegenNative() {
        try {
            val tag = NativeImageGen.nativePing()
            Log.i(TAG, "imagegen native loaded: ping=$tag")
        } catch (t: Throwable) {
            Log.w(TAG, "imagegen native not available", t)
            return
        }
        // FastRPC loads the per-Hexagon-rev Skel onto the DSP using a real
        // filesystem path; point it at our nativeLibraryDir, where AGP's
        // legacy-packaging extracts libQnnHtpV73Skel.so on install.
        val nativeLibDir = applicationInfo.nativeLibraryDir
        try {
            if (!NativeImageGen.nativeSetAdspLibraryPath(nativeLibDir)) {
                Log.w(TAG, "nativeSetAdspLibraryPath returned false for $nativeLibDir")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "nativeSetAdspLibraryPath threw", t)
        }
        probeQnnInspectIfBinaryPresent()
    }

    /**
     * Phase 4a verification: if a QNN binary from the bundle is on device,
     * load + inspect its metadata. Tries unet.bin first, then the VAE
     * binaries (smaller; useful while iterating). Skips silently if none
     * are present.
     *
     * Phase 4b verification: if `clip_v2.mnn` + `tokenizer.json` are present,
     * load the MNN model, run a forward pass on "a cat", log the first 8
     * output floats.
     */
    private fun probeQnnInspectIfBinaryPresent() {
        val modelDir = File(File(filesDir, "models"), DEFAULT_DIFFUSION_DIR_NAME)
        if (!modelDir.isDirectory) {
            Log.i(TAG, "no model dir at $modelDir — skip inspect (push a bundle to verify)")
            return
        }

        // QNN binaries.
        val qnnCandidates = listOf("unet.bin", "vae_decoder.bin", "vae_encoder.bin")
        val qnnBin = qnnCandidates.map { File(modelDir, it) }.firstOrNull { it.isFile }
        if (qnnBin != null) {
            try {
                NativeImageGen.nativeInspectQnnBinary(qnnBin.absolutePath)
                    .lineSequence().forEach { Log.i(TAG, it) }
            } catch (t: Throwable) {
                Log.w(TAG, "QNN inspect failed for ${qnnBin.name}", t)
            }
        } else {
            Log.i(TAG, "no QNN binary in $modelDir — skipping QNN probe")
        }

        // MNN text encoder + tokenizer.
        val clip = File(modelDir, "clip_v2.mnn")
        val tok  = File(modelDir, "tokenizer.json")
        if (!clip.isFile) {
            Log.i(TAG, "no clip_v2.mnn at $modelDir — skipping MNN probe")
            return
        }
        try {
            NativeImageGen.nativeInspectMnnModel(clip.absolutePath)
                .lineSequence().forEach { Log.i(TAG, it) }
        } catch (t: Throwable) {
            Log.w(TAG, "MNN inspect failed", t)
            return
        }
        val tokenEmb = File(modelDir, "token_emb.bin")
        val posEmb   = File(modelDir, "pos_emb.bin")
        if (!tok.isFile || !tokenEmb.isFile || !posEmb.isFile) {
            Log.i(TAG, "MNN forward probe needs tokenizer.json + token_emb.bin + pos_emb.bin")
            return
        }
        try {
            val out = NativeImageGen.nativeRunMnnTextEncode(modelDir.absolutePath, "a cat")
            if (out == null || out.isEmpty()) {
                Log.w(TAG, "MNN text-encode returned null/empty")
            } else {
                val first8 = out.take(8).joinToString(", ") { "%.4f".format(it) }
                Log.i(TAG, "clip_v2.mnn 'a cat' first8 = [$first8]  totalFloats=${out.size}")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "MNN text-encode failed", t)
        }

        // Phase 7 probe: when the heavy QNN binary is present, drive the full
        // pipeline (text encode + UNet diffusion + VAE decode + PNG encode)
        // and write the result to filesDir/sd-debug.png. Iters are kept low
        // (8) to keep boot time bounded; widget calls pass a higher iter count
        // for production-quality output.
        val unet = File(modelDir, "unet.bin")
        val vae  = File(modelDir, "vae_decoder.bin")
        if (!unet.isFile || !vae.isFile) {
            Log.i(TAG, "missing unet.bin or vae_decoder.bin at $modelDir — skip diffusion probe")
            return
        }
        try {
            val png = NativeImageGen.nativeRunDiffusionPng(
                modelDir.absolutePath,
                /*prompt=*/"a cat",
                /*iters=*/8,
                /*seed=*/42L,
            )
            if (png == null || png.isEmpty()) {
                Log.w(TAG, "diffusion-to-png probe returned empty — see logcat tags 'diffusion' / 'imagegen'")
            } else {
                val out = File(filesDir, "sd-debug.png")
                out.writeBytes(png)
                Log.i(TAG, "diffusion-to-png probe OK: pngBytes=${png.size} → ${out.absolutePath}")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "diffusion-to-png probe failed", t)
        }
    }

    companion object {
        const val CHANNEL_ID = "localai_inference"
        private const val TAG = "LocalAiApp"
    }
}
