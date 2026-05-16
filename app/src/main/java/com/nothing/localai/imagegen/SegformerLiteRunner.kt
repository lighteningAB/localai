package com.nothing.localai.imagegen

import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.io.path.Path

private const val TAG = "SegformerLiteRunner"

/**
 * LiteRT (TFLite) wrapper for mattmdjaga/segformer_b2_clothes, compiled to a
 * .tflite by AI Hub. Replaces the QNN path (which fails 0x138d on this device).
 *
 * Input: 1×3×512×512 RGB fp32, ImageNet-normalized. Output: 1×18×H×W class
 * logits where (H,W) is whichever the converted model emits (typically
 * 128×128). We argmax to a uint8 class map sized H*W.
 *
 * Input layout is detected from the interpreter's input tensor shape — Hub's
 * TFLite export may keep NCHW from PyTorch or transpose to NHWC. The runner
 * adapts.
 */
class SegformerLiteRunner(modelFile: File) : AutoCloseable {

    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val outputShape: IntArray

    val inputIsNhwc: Boolean
    val numClasses: Int
    val outH: Int
    val outW: Int

    init {
        val buf = mapFile(modelFile)
        val opts = Interpreter.Options().apply {
            // CPU only for v1 — GPU delegate often falls back for SegFormer
            // attention ops and can produce different numerics. Tune in Phase 6.
            setNumThreads(4)
        }
        interpreter = Interpreter(buf, opts)
        inputShape = interpreter.getInputTensor(0).shape()
        outputShape = interpreter.getOutputTensor(0).shape()

        require(inputShape.size == 4) { "expected 4D input, got ${inputShape.toList()}" }
        // Input: NCHW=[1,3,512,512] or NHWC=[1,512,512,3].
        inputIsNhwc = when {
            inputShape[1] == 3 -> false
            inputShape[3] == 3 -> true
            else -> error("input shape ${inputShape.toList()} not classifiable as 3-channel 4D")
        }

        require(outputShape.size == 4) { "expected 4D output, got ${outputShape.toList()}" }
        when {
            outputShape[1] == 18 -> { numClasses = 18; outH = outputShape[2]; outW = outputShape[3] }
            outputShape[3] == 18 -> { numClasses = 18; outH = outputShape[1]; outW = outputShape[2] }
            else -> error("output shape ${outputShape.toList()} not classifiable as 18-class 4D")
        }
        Log.i(TAG, "model loaded: in=${inputShape.toList()} (${if (inputIsNhwc) "NHWC" else "NCHW"}) " +
                "out=${outputShape.toList()} numClasses=$numClasses outH=$outH outW=$outW")
    }

    /**
     * Decode + normalize + run + argmax. Returns a flat byte[outH*outW] of
     * SegFormer class IDs (0..17). Caller applies mask_ops chain to translate
     * to the final inpaint mask. Resizes/center-crops to 512×512.
     */
    fun classmapFromBitmap(raw: Bitmap): ByteArray {
        val rgb = decodeAndNormalize(raw)
        return classmapFromCHW(rgb)
    }

    /**
     * As [classmapFromBitmap], but the caller supplies an ImageNet-normalized
     * 3×512×512 fp32 CHW row-major buffer. Used by the outfit-swap pipeline
     * which already has the float buffer for the VAE encoder downstream.
     */
    fun classmapFromCHW(normalizedChw: FloatArray): ByteArray {
        require(normalizedChw.size == 3 * 512 * 512) {
            "expected 3*512*512 floats, got ${normalizedChw.size}"
        }

        val inputBuf = ByteBuffer
            .allocateDirect(normalizedChw.size * 4)
            .order(ByteOrder.nativeOrder())
        val asFloat = inputBuf.asFloatBuffer()
        if (inputIsNhwc) {
            // CHW → HWC interleave.
            val plane = 512 * 512
            for (i in 0 until plane) {
                asFloat.put(normalizedChw[i])
                asFloat.put(normalizedChw[plane + i])
                asFloat.put(normalizedChw[2 * plane + i])
            }
        } else {
            asFloat.put(normalizedChw)
        }
        inputBuf.rewind()

        val outputBuf = ByteBuffer
            .allocateDirect(numClasses * outH * outW * 4)
            .order(ByteOrder.nativeOrder())
        outputBuf.rewind()
        interpreter.run(inputBuf, outputBuf)
        outputBuf.rewind()

        return argmax(outputBuf.asFloatBuffer(), numClasses, outH, outW, outIsNhwc())
    }

    fun outIsNhwc(): Boolean = outputShape[3] == numClasses

    override fun close() {
        interpreter.close()
    }

    companion object {
        // Same ImageNet stats SegformerProbe used for the QNN path.
        @JvmStatic
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        @JvmStatic
        val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
        const val SIZE = 512

        @JvmStatic
        fun decodeAndNormalize(raw: Bitmap): FloatArray {
            val side = minOf(raw.width, raw.height)
            val cx = (raw.width - side) / 2
            val cy = (raw.height - side) / 2
            val square = if (raw.width == side && raw.height == side) raw
            else Bitmap.createBitmap(raw, cx, cy, side, side)
            val resized = if (square.width == SIZE && square.height == SIZE) square
            else Bitmap.createScaledBitmap(square, SIZE, SIZE, /*filter=*/true)
            if (raw !== square && square !== resized) square.recycle()

            val pixels = IntArray(SIZE * SIZE)
            resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
            val out = FloatArray(3 * SIZE * SIZE)
            val plane = SIZE * SIZE
            for (i in pixels.indices) {
                val px = pixels[i]
                val r = ((px shr 16) and 0xFF) / 255f
                val g = ((px shr 8)  and 0xFF) / 255f
                val b = ( px         and 0xFF) / 255f
                out[i]             = (r - MEAN[0]) / STD[0]
                out[plane + i]     = (g - MEAN[1]) / STD[1]
                out[2 * plane + i] = (b - MEAN[2]) / STD[2]
            }
            if (resized !== raw) resized.recycle()
            return out
        }

        private fun mapFile(f: File): MappedByteBuffer {
            FileChannel.open(Path(f.absolutePath)).use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size())
            }
        }

        private fun argmax(
            logits: java.nio.FloatBuffer,
            C: Int, H: Int, W: Int, isNhwc: Boolean,
        ): ByteArray {
            val out = ByteArray(H * W)
            if (isNhwc) {
                val perPixel = FloatArray(C)
                for (i in 0 until H * W) {
                    logits.get(perPixel)
                    var bestI = 0; var bestV = perPixel[0]
                    for (c in 1 until C) if (perPixel[c] > bestV) { bestV = perPixel[c]; bestI = c }
                    out[i] = bestI.toByte()
                }
            } else {
                // NCHW: pull into a local array, then argmax over C-stride.
                val plane = H * W
                val all = FloatArray(C * plane)
                logits.get(all)
                for (i in 0 until plane) {
                    var bestI = 0; var bestV = all[i]
                    for (c in 1 until C) {
                        val v = all[c * plane + i]
                        if (v > bestV) { bestV = v; bestI = c }
                    }
                    out[i] = bestI.toByte()
                }
            }
            return out
        }
    }
}
