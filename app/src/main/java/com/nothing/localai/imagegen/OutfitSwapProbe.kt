package com.nothing.localai.imagegen

import android.content.Context
import android.os.ParcelFileDescriptor
import android.util.Log
import com.nothing.localai.IImageGenCallback
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

private const val TAG = "OutfitSwapProbe"

/**
 * Phase 0 end-to-end probe: drive the full outfit-swap pipeline from the
 * boot path, bypassing the Aiwidget AIDL layer. Confirms SegFormer → MI-GAN
 * → SD txt2img → composite all chain together on this device.
 *
 * Reads the test JPEG already in `filesDir/segformer-probe/` (pushed via
 * run-as for the SegFormer / MI-GAN probes) and writes the final composite
 * to `filesDir/segformer-probe/outfit-swap-debug.png`.
 *
 * Skips silently if either the xororz bundle or the LiteRT model files are
 * missing.
 */
object OutfitSwapProbe {

    fun run(
        ctx: Context,
        prompt: String = "red plaid flannel shirt",
        garmentSpec: String = "upper-clothes",
        iterations: Int = 12,
        seed: Long = 42L,
        timeoutMs: Long = 180_000L,
    ): Boolean {
        val probeDir = File(ctx.filesDir, "segformer-probe")
        val jpg = File(probeDir, "test_portrait.jpg")
        if (!jpg.isFile) {
            Log.i(TAG, "skip — push test_portrait.jpg via run-as first")
            return false
        }
        val runner = OutfitSwapRunner(ctx)
        if (!runner.isReady()) {
            Log.i(TAG, "skip — runner not ready (missing bundle or tflite models)")
            return false
        }
        val outPng = File(probeDir, "outfit-swap-debug.png")

        val latch = CountDownLatch(1)
        var resultOk = false
        var resultErr: String? = null

        val cb = object : IImageGenCallback.Stub() {
            override fun onStep(rid: String, step: Int, total: Int) {
                Log.i(TAG, "step $step/$total")
            }
            override fun onStage(rid: String, stage: String) {
                Log.i(TAG, "stage: $stage")
            }
            override fun onResult(
                rid: String, pngFd: ParcelFileDescriptor, width: Int, height: Int,
            ) {
                try {
                    ParcelFileDescriptor.AutoCloseInputStream(pngFd).use { ins ->
                        outPng.outputStream().use { os -> ins.copyTo(os) }
                    }
                    Log.i(TAG, "result $width×$height → ${outPng.absolutePath}")
                    resultOk = true
                } catch (t: Throwable) {
                    Log.e(TAG, "write result failed", t)
                    resultErr = t.message
                } finally {
                    latch.countDown()
                }
            }
            override fun onError(rid: String, code: String, msg: String) {
                Log.e(TAG, "error $code: $msg")
                resultErr = "$code: $msg"
                latch.countDown()
            }
        }

        // Open the JPEG as a PFD that the runner can read.
        val pfd = ParcelFileDescriptor.open(jpg, ParcelFileDescriptor.MODE_READ_ONLY)
        runner.generate(pfd, prompt, garmentSpec, iterations, seed, cb)
        if (!latch.await(timeoutMs, TimeUnit.MILLISECONDS)) {
            Log.w(TAG, "timed out after ${timeoutMs}ms")
            return false
        }
        runner.close()
        return resultOk
    }
}
