package com.nothing.localai.imagegen

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.File

/**
 * Phase 0c+d (outfit-swap) hardware-verification helper.
 *
 * Loads a Hub-compiled TFLite SegFormer via LiteRT, runs it on a JPEG, and
 * writes a colored class-map PNG so the result can be visually inspected.
 *
 * Usage from [com.nothing.localai.LocalAiApp]'s boot probe:
 *
 * ```kotlin
 * val ok = SegformerProbe.run(
 *     tfliteModelPath = "${context.filesDir}/segformer-probe/segformer_b2_clothes.tflite",
 *     inputJpegPath   = "${context.filesDir}/segformer-probe/test_portrait.jpg",
 *     outputPngPath   = "${context.filesDir}/segformer-probe/segformer-debug.png",
 * )
 * ```
 *
 * Push files via `adb shell run-as com.nothing.localai.debug` since the app
 * sandbox is the only path the native code can read without SELinux denials.
 */
object SegformerProbe {

    private const val TAG = "SegformerProbe"

    // 18-color palette for ATR/iMaterialist classes (mirror of segformer.cpp).
    private val PALETTE = arrayOf(
        intArrayOf(  0,   0,   0),  // 0  Background
        intArrayOf(128,   0,   0),  // 1  Hat
        intArrayOf(255, 200,   0),  // 2  Hair
        intArrayOf(  0, 200, 255),  // 3  Sunglasses
        intArrayOf(255,  50,  50),  // 4  Upper-clothes
        intArrayOf(  0, 200,   0),  // 5  Skirt
        intArrayOf( 50,  50, 220),  // 6  Pants
        intArrayOf(255, 100, 200),  // 7  Dress
        intArrayOf(200, 200,   0),  // 8  Belt
        intArrayOf(120,  60,  20),  // 9  Left-shoe
        intArrayOf(180,  90,  30),  // 10 Right-shoe
        intArrayOf(255, 220, 180),  // 11 Face
        intArrayOf(180, 100, 100),  // 12 Left-leg
        intArrayOf(200, 120, 120),  // 13 Right-leg
        intArrayOf(100, 180, 200),  // 14 Left-arm
        intArrayOf(130, 200, 220),  // 15 Right-arm
        intArrayOf(150,  80, 200),  // 16 Bag
        intArrayOf(255, 180,  60),  // 17 Scarf
    )

    /**
     * Decode [inputJpegPath], run the LiteRT SegFormer at [tfliteModelPath],
     * and write the colored class-map PNG to [outputPngPath].
     */
    fun run(
        tfliteModelPath: String,
        inputJpegPath: String,
        outputPngPath: String,
    ): Boolean {
        val modelFile = File(tfliteModelPath)
        if (!modelFile.isFile) {
            Log.e(TAG, "model missing: $tfliteModelPath")
            return false
        }
        val inputFile = File(inputJpegPath)
        if (!inputFile.isFile) {
            Log.e(TAG, "input image missing: $inputJpegPath")
            return false
        }
        val raw = BitmapFactory.decodeFile(inputFile.absolutePath)
            ?: run { Log.e(TAG, "decode failed for $inputJpegPath"); return false }

        val t0 = System.currentTimeMillis()
        val runner = SegformerLiteRunner(modelFile)
        val tLoad = System.currentTimeMillis()
        val classmap = runner.use { it.classmapFromBitmap(raw) }
        val tInfer = System.currentTimeMillis()

        val outH = runner.outH
        val outW = runner.outW
        Log.i(TAG, "inference ok: ${outH}x${outW} classes=${runner.numClasses} " +
                "load=${tLoad - t0}ms infer=${tInfer - tLoad}ms")
        // Class histogram for sanity.
        val hist = IntArray(18)
        for (b in classmap) {
            val c = b.toInt() and 0xFF
            if (c < 18) hist[c]++
        }
        Log.i(TAG, "class histogram = ${hist.toList()}")

        // Colorize at the model's output resolution (typically 128×128) — keep
        // small for the debug PNG; production pipeline upsamples in mask_ops.
        val rgb = ByteArray(outH * outW * 3)
        for (i in classmap.indices) {
            val c = classmap[i].toInt() and 0xFF
            val color = if (c < 18) PALETTE[c] else intArrayOf(255, 0, 255)
            rgb[i * 3 + 0] = color[0].toByte()
            rgb[i * 3 + 1] = color[1].toByte()
            rgb[i * 3 + 2] = color[2].toByte()
        }
        val bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(outW * outH)
        for (i in pixels.indices) {
            val r = rgb[i * 3 + 0].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
        val outFile = File(outputPngPath).apply { parentFile?.mkdirs() }
        outFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bmp.recycle()
        Log.i(TAG, "wrote $outputPngPath")
        return true
    }
}
