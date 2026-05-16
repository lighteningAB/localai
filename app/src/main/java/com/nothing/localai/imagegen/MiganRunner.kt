package com.nothing.localai.imagegen

import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.io.path.Path

private const val TAG = "MiganRunner"

/**
 * LiteRT wrapper around picsart-ai-research/mi-gan (`migan.onnx` core,
 * Hub-compiled to `migan.tflite`). Single forward pass — no diffusion loop.
 *
 * Input:
 *   image — uint8 [1, 3, 512, 512] CHW RGB
 *   mask  — uint8 [1, 1, 512, 512]  (0 = pixel will be filled, 255 = keep)
 * Output:
 *   uint8 [1, 3, 512, 512] CHW — the input image with the masked region
 *   context-filled (skin/background extrapolation, no text conditioning).
 *
 * Used as Stage 2 of the outfit-swap hybrid (PLAN-OUTFIT-SWAP.md §13.2):
 * SegFormer mask → MI-GAN erase → base UNet txt2img on prompt → composite.
 */
class MiganRunner(modelFile: File) : AutoCloseable {

    private val interpreter: Interpreter

    init {
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(mapFile(modelFile), opts)
        val inA = interpreter.getInputTensor(0).shape().toList()
        val inB = interpreter.getInputTensor(1).shape().toList()
        val out = interpreter.getOutputTensor(0).shape().toList()
        Log.i(TAG, "model loaded: in0=$inA in1=$inB out=$out")
    }

    /**
     * Erase the masked region of [rgb] and fill via the MI-GAN context-fill.
     * Both inputs are 512×512. `mask` length must be 512×512 with the MI-GAN
     * convention (`0` = erase, `255` = keep). Returns a new 512×512 Bitmap.
     */
    fun erase(rgb: Bitmap, mask: ByteArray): Bitmap {
        require(rgb.width == SIZE && rgb.height == SIZE) {
            "rgb must be 512×512, got ${rgb.width}×${rgb.height}"
        }
        require(mask.size == SIZE * SIZE) {
            "mask must be ${SIZE * SIZE} bytes, got ${mask.size}"
        }

        val imgBytes = ByteArray(3 * SIZE * SIZE)
        val pixels = IntArray(SIZE * SIZE)
        rgb.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val px = pixels[i]
            imgBytes[i]              = ((px shr 16) and 0xFF).toByte()
            imgBytes[plane + i]      = ((px shr 8)  and 0xFF).toByte()
            imgBytes[2 * plane + i]  = ( px         and 0xFF).toByte()
        }

        val imageBuf = ByteBuffer.allocateDirect(imgBytes.size).order(ByteOrder.nativeOrder())
        imageBuf.put(imgBytes); imageBuf.rewind()
        val maskBuf = ByteBuffer.allocateDirect(mask.size).order(ByteOrder.nativeOrder())
        maskBuf.put(mask); maskBuf.rewind()
        val outBuf = ByteBuffer.allocateDirect(3 * SIZE * SIZE).order(ByteOrder.nativeOrder())

        val inputs = arrayOf<Any>(imageBuf, maskBuf)
        val outputs = mapOf(0 to outBuf)
        interpreter.runForMultipleInputsOutputs(inputs, outputs)

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
        return Bitmap.createBitmap(outPixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        const val SIZE = 512

        private fun mapFile(f: File): MappedByteBuffer {
            FileChannel.open(Path(f.absolutePath)).use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size())
            }
        }
    }
}
