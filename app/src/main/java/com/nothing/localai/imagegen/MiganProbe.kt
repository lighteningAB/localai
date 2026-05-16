package com.nothing.localai.imagegen

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.io.path.Path

private const val TAG = "MiganProbe"

/**
 * Phase 1 verification helper: load `migan.tflite` via LiteRT, run a single
 * erase over the SegFormer-derived mask, and write the result PNG so we can
 * visually verify MI-GAN is producing a plausible context-fill on this device.
 *
 * Pipeline:
 *   1. Load `segformer_b2_clothes.tflite` and `migan.tflite` side by side.
 *   2. Decode the input JPEG, resize to 512×512.
 *   3. Run SegFormer → classmap (128×128 → upsample to 512×512).
 *   4. Build a binary mask: 0 where classmap ∈ {upper-clothes, dress},
 *      255 elsewhere (MI-GAN convention: 0 = masked, 255 = known).
 *   5. Run MI-GAN inference: image (uint8 1×3×512×512) + mask (uint8 1×1×512×512)
 *      → result (uint8 1×3×512×512).
 *   6. Encode result PNG.
 *
 * Inputs and outputs are all uint8 — MI-GAN does its own normalization.
 */
object MiganProbe {

    private const val SIZE = 512

    // SegFormer classes to erase. v1 hero classes for outfit-swap.
    private val ERASE_CLASSES = setOf(4, 5, 6, 7)  // upper-clothes, skirt, pants, dress

    fun run(
        miganModelPath: String,
        segformerModelPath: String,
        inputJpegPath: String,
        outputPngPath: String,
    ): Boolean {
        val miganFile = File(miganModelPath)
        val segFile = File(segformerModelPath)
        val jpgFile = File(inputJpegPath)
        if (!miganFile.isFile || !segFile.isFile || !jpgFile.isFile) {
            Log.e(TAG, "missing inputs: migan=${miganFile.exists()} seg=${segFile.exists()} jpg=${jpgFile.exists()}")
            return false
        }

        val raw = BitmapFactory.decodeFile(jpgFile.absolutePath)
            ?: run { Log.e(TAG, "decode failed"); return false }
        val resized = resizeTo512(raw)
        raw.recycle()

        // SegFormer → classmap (128×128).
        val tSegStart = System.currentTimeMillis()
        val classmapLow: ByteArray
        val segH: Int
        val segW: Int
        SegformerLiteRunner(segFile).use { seg ->
            classmapLow = seg.classmapFromBitmap(resized)
            segH = seg.outH
            segW = seg.outW
        }
        Log.i(TAG, "segformer ${segH}×${segW} done in ${System.currentTimeMillis() - tSegStart}ms")

        // Upsample classmap to 512×512 (nearest) and turn into a 0/255 mask
        // matching MI-GAN's convention: 0 = masked, 255 = keep.
        val mask512 = ByteArray(SIZE * SIZE)
        for (y in 0 until SIZE) {
            val sy = (y * segH / SIZE).coerceAtMost(segH - 1)
            for (x in 0 until SIZE) {
                val sx = (x * segW / SIZE).coerceAtMost(segW - 1)
                val cls = classmapLow[sy * segW + sx].toInt() and 0xFF
                mask512[y * SIZE + x] = if (cls in ERASE_CLASSES) 0.toByte() else 0xFF.toByte()
            }
        }
        val maskedCount = mask512.count { (it.toInt() and 0xFF) == 0 }
        Log.i(TAG, "mask: ${maskedCount}/${mask512.size} pixels masked")
        if (maskedCount == 0) {
            Log.w(TAG, "nothing to erase — pick an input with garment classes")
            return false
        }

        // Build MI-GAN input buffers (uint8 CHW).
        val imgBytes = ByteArray(3 * SIZE * SIZE)
        val pixels = IntArray(SIZE * SIZE)
        resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val px = pixels[i]
            imgBytes[i]              = ((px shr 16) and 0xFF).toByte()
            imgBytes[plane + i]      = ((px shr 8)  and 0xFF).toByte()
            imgBytes[2 * plane + i]  = ( px         and 0xFF).toByte()
        }
        resized.recycle()

        val imageBuf = ByteBuffer.allocateDirect(imgBytes.size).order(ByteOrder.nativeOrder())
        imageBuf.put(imgBytes); imageBuf.rewind()
        val maskBuf = ByteBuffer.allocateDirect(mask512.size).order(ByteOrder.nativeOrder())
        maskBuf.put(mask512); maskBuf.rewind()
        val outBuf = ByteBuffer.allocateDirect(3 * SIZE * SIZE).order(ByteOrder.nativeOrder())

        // Run MI-GAN.
        val tInferStart = System.currentTimeMillis()
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        val interp = Interpreter(mapFile(miganFile), opts)
        Log.i(TAG, "migan input[0]=${interp.getInputTensor(0).shape().toList()} " +
                "input[1]=${interp.getInputTensor(1).shape().toList()} " +
                "output=${interp.getOutputTensor(0).shape().toList()}")
        val inputs = arrayOf<Any>(imageBuf, maskBuf)
        val outputs = mapOf(0 to outBuf)
        interp.runForMultipleInputsOutputs(inputs, outputs)
        interp.close()
        Log.i(TAG, "migan inference done in ${System.currentTimeMillis() - tInferStart}ms")

        // Output is uint8 CHW. Convert to bitmap.
        outBuf.rewind()
        val outBytes = ByteArray(3 * SIZE * SIZE)
        outBuf.get(outBytes)
        val outPixels = IntArray(SIZE * SIZE)
        for (i in 0 until plane) {
            val r = outBytes[i].toInt() and 0xFF
            val g = outBytes[plane + i].toInt() and 0xFF
            val b = outBytes[2 * plane + i].toInt() and 0xFF
            outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        bmp.setPixels(outPixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val outFile = File(outputPngPath).apply { parentFile?.mkdirs() }
        outFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bmp.recycle()
        Log.i(TAG, "wrote $outputPngPath")
        return true
    }

    private fun resizeTo512(raw: Bitmap): Bitmap {
        val side = minOf(raw.width, raw.height)
        val cx = (raw.width - side) / 2
        val cy = (raw.height - side) / 2
        val square = if (raw.width == side && raw.height == side) raw
        else Bitmap.createBitmap(raw, cx, cy, side, side)
        val resized = if (square.width == SIZE && square.height == SIZE) square
        else Bitmap.createScaledBitmap(square, SIZE, SIZE, /*filter=*/true)
        if (raw !== square && square !== resized) square.recycle()
        return resized
    }

    private fun mapFile(f: File): java.nio.MappedByteBuffer {
        FileChannel.open(Path(f.absolutePath)).use { ch ->
            return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size())
        }
    }
}
