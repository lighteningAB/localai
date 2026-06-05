package com.nothing.localai

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.nothing.localai.imagegen.DEFAULT_DIFFUSION_DIR_NAME
import com.nothing.localai.imagegen.MiganProbe
import com.nothing.localai.imagegen.NativeImageGen
import com.nothing.localai.imagegen.OutfitSwapProbe
import com.nothing.localai.imagegen.SegformerProbe
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
        probeSdxlVerifyIfPresent()
    }

    /**
     * One-shot SDXL bundle compatibility probe. If a UNet context binary has
     * been pushed to `<filesDir>/models/sd-xl-verify/unet.bin` (see
     * `scripts/probe-sdxl-bundle.sh`), run init + inspectBinary + instantiate
     * against it and log the multi-line PASS/FAIL report.
     *
     * The decisive test is the `instantiate` step — it calls
     * `QnnContext_createFromBinary` against the live HTP backend and surfaces
     * incompatibility (e.g. an _8gen3 / V75-targeted SDXL binary on a V73
     * device) as `rc=0x...`. The report ends with a `VERDICT:` line for an
     * unambiguous grep.
     *
     * To re-run, push a new binary into the same path; to disable, delete
     * the directory.
     */
    private fun probeSdxlVerifyIfPresent() {
        val verifyDir = File(File(filesDir, "models"), "sd-xl-verify")
        val unet = File(verifyDir, "unet.bin")
        if (!unet.isFile) {
            Log.i(TAG, "no sd-xl-verify/unet.bin at $verifyDir — skip SDXL probe " +
                "(push with scripts/probe-sdxl-bundle.sh)")
            return
        }
        Log.i(TAG, "SDXL probe: ${unet.absolutePath} (${unet.length()} bytes)")
        try {
            val report = NativeImageGen.nativeProbeQnnBinaryLoad(unet.absolutePath)
            report.lineSequence().forEach { Log.i(TAG, "sdxl-probe: $it") }
        } catch (t: Throwable) {
            Log.w(TAG, "SDXL probe threw", t)
        }
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

        // Outfit-swap Phase 0c+d probe: if a SegFormer binary and a test image
        // are present at /data/local/tmp/ (push them via adb), run the model
        // and dump a colored mask PNG. Confirms QAIRT 2.45→2.46 forward compat.
        probeSegformerIfPresent()
    }

    /**
     * Phase 0c+d outfit-swap probe. Looks for the AI Hub-compiled SegFormer
     * binary + a test JPEG inside this app's *internal* `filesDir`, which is
     * accessible from native code without SELinux issues. Push via run-as on
     * a debug-signed build:
     * ```
     *   adb shell run-as com.nothing.localai.debug \
     *     mkdir -p files/segformer-probe
     *   adb push segformer_b2_clothes.bin /sdcard/...staging path...
     *   adb shell 'cat /sdcard/... | run-as com.nothing.localai.debug \
     *     tee files/segformer-probe/segformer_b2_clothes.bin > /dev/null'
     * ```
     * The result PNG lands alongside as `segformer-debug.png`.
     * See PLAN-OUTFIT-SWAP.md §10 Phase 0.
     */
    private fun probeSegformerIfPresent() {
        val probeDir = File(filesDir, "segformer-probe").apply { mkdirs() }
        val model = File(probeDir, "segformer_b2_clothes.tflite")
        val jpg = File(probeDir, "test_portrait.jpg")
        Log.i(TAG, "segformer probe: looking in ${probeDir.absolutePath}")
        Log.i(TAG, "  model exists=${model.exists()} size=${if (model.exists()) model.length() else 0}")
        Log.i(TAG, "  jpg exists=${jpg.exists()} size=${if (jpg.exists()) jpg.length() else 0}")
        if (!model.isFile || !jpg.isFile) {
            Log.i(TAG, "segformer probe: skip (use run-as to populate $probeDir)")
            return
        }
        try {
            val ok = SegformerProbe.run(
                tfliteModelPath = model.absolutePath,
                inputJpegPath   = jpg.absolutePath,
                outputPngPath   = File(probeDir, "segformer-debug.png").absolutePath,
            )
            Log.i(TAG, "segformer probe: ok=$ok")
        } catch (t: Throwable) {
            Log.w(TAG, "segformer probe threw", t)
        }

        val migan = File(probeDir, "migan.tflite")
        if (migan.isFile) {
            try {
                val ok = MiganProbe.run(
                    miganModelPath     = migan.absolutePath,
                    segformerModelPath = model.absolutePath,
                    inputJpegPath      = jpg.absolutePath,
                    outputPngPath      = File(probeDir, "migan-debug.png").absolutePath,
                )
                Log.i(TAG, "migan probe: ok=$ok")
            } catch (t: Throwable) {
                Log.w(TAG, "migan probe threw", t)
            }
        }

        // Full end-to-end outfit-swap probe.
        try {
            val ok = OutfitSwapProbe.run(this)
            Log.i(TAG, "outfit-swap probe: ok=$ok")
        } catch (t: Throwable) {
            Log.w(TAG, "outfit-swap probe threw", t)
        }
    }

    companion object {
        const val CHANNEL_ID = "localai_inference"
        private const val TAG = "LocalAiApp"
    }
}
