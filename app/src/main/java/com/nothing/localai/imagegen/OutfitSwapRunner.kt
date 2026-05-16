package com.nothing.localai.imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.ParcelFileDescriptor
import android.os.RemoteException
import android.util.Log
import com.nothing.localai.IImageGenCallback
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

private const val TAG = "OutfitSwapRunner"

/**
 * Outfit-swap pipeline runner — Kotlin/LiteRT hybrid (see PLAN-OUTFIT-SWAP.md).
 *
 * The original plan called for a single QNN-compiled SD 1.5 inpaint UNet to
 * drive the whole pipeline. That model wouldn't compile to a loadable artifact
 * via AI Hub (QNN blob format incompatibility, TFLite flatbuffer size cap), so
 * v1 uses a three-step hybrid that combines models that DO load on this device:
 *
 *   1. SegFormer-B2-Clothes  (LiteRT, ~2.5s)
 *      → 18-class garment classmap → binary mask via [parseGarmentSpec].
 *   2. MI-GAN context-fill   (LiteRT, ~1.6s)
 *      → erases the garment region, leaving a "clean" portrait with skin/
 *        background extrapolation where the clothes were.
 *   3. SD 1.5 base UNet      (existing xororz QNN bundle, ~10s @ 8 iters)
 *      → txt2img with a pose-anchoring prompt prefix that re-paints the full
 *        scene in the user's described outfit.
 *   4. Composite             (Kotlin)
 *      → Gaussian-feathered alpha blend: erased outside the mask
 *        (preserves face/hair/background); generated inside the mask
 *        (new garment).
 *
 * Identity preservation is weaker than a true SD-inpaint pipeline would
 * provide — the txt2img pass doesn't know about the original face. Future
 * work (Phase 6) replaces step 3 with an img2img variant of the base UNet
 * starting from the MI-GAN-erased latent.
 *
 * Model file layout:
 * ```
 * filesDir/models/sd-v15-xororz/                  (existing — base UNet + VAE + CLIP)
 * filesDir/segformer-probe/segformer_b2_clothes.tflite   (push via run-as)
 * filesDir/segformer-probe/migan.tflite                  (push via run-as)
 * ```
 */
class OutfitSwapRunner(
    private val ctx: Context,
    private val bundleDirName: String = DEFAULT_DIFFUSION_DIR_NAME,
) {

    private val jobs = ConcurrentHashMap<String, Job>()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val bundleDir: File
        get() = File(File(ctx.filesDir, "models"), bundleDirName)

    private val probeDir: File
        get() = File(ctx.filesDir, "segformer-probe")

    private val segformerFile: File
        get() = File(probeDir, "segformer_b2_clothes.tflite")

    private val miganFile: File
        get() = File(probeDir, "migan.tflite")

    fun isReady(): Boolean {
        val xororzReady = listOf(
            "tokenizer.json", "clip_v2.mnn", "pos_emb.bin", "token_emb.bin",
            "vae_encoder.bin", "vae_decoder.bin", "unet.bin",
        ).all { File(bundleDir, it).exists() }
        return xororzReady && segformerFile.exists() && miganFile.exists()
    }

    fun generate(
        inputPng: ParcelFileDescriptor,
        prompt: String,
        garmentSpec: String,
        iterations: Int,
        seed: Long,
        cb: IImageGenCallback,
    ): String {
        val requestId = UUID.randomUUID().toString()
        val selectedClasses = parseGarmentSpec(garmentSpec)
        if (selectedClasses == null) {
            safe { cb.onError(requestId, "BAD_GARMENT_SPEC",
                "unrecognized garmentSpec='$garmentSpec'") }
            runCatching { inputPng.close() }
            return requestId
        }
        if (!isReady()) {
            safe {
                cb.onError(requestId, "BUNDLE_MISSING",
                    "missing one of: $bundleDir, $segformerFile, $miganFile")
            }
            runCatching { inputPng.close() }
            return requestId
        }

        val job = scope.launch {
            try {
                val pngBytes = FileInputStream(inputPng.fileDescriptor).use { it.readBytes() }
                runCatching { inputPng.close() }

                // ---- 1. Decode + resize input to 512×512 ----
                val photo512 = decodeAndResize512(pngBytes)
                    ?: run {
                        safe { cb.onError(requestId, "DECODE_FAILED", "couldn't decode input PNG") }
                        return@launch
                    }

                // ---- 2. SegFormer → garment mask ----
                safe { cb.onStage(requestId, "segmenting") }
                val classmap: ByteArray
                val segH: Int
                val segW: Int
                SegformerLiteRunner(segformerFile).use { seg ->
                    classmap = seg.classmapFromBitmap(photo512)
                    segH = seg.outH
                    segW = seg.outW
                }
                val mask512 = classmapToMask512(classmap, segH, segW, selectedClasses)
                val masked = mask512.count { (it.toInt() and 0xFF) == 0 }
                Log.i(TAG, "mask: $masked / ${mask512.size} pixels masked")
                if (masked < MIN_MASK_PIXELS) {
                    safe {
                        cb.onError(requestId, "NO_GARMENT_DETECTED",
                            "SegFormer found no garment of type '$garmentSpec' " +
                                "in this photo (masked $masked < $MIN_MASK_PIXELS pixels)")
                    }
                    return@launch
                }

                // ---- 3. MI-GAN erase ----
                safe { cb.onStage(requestId, "encoding") }  // reuse AIDL stage label
                val erased512 = MiganRunner(miganFile).use { it.erase(photo512, mask512) }
                Log.i(TAG, "migan erased")

                // ---- 4. SD 1.5 img2img seeded from the MI-GAN-erased portrait ----
                //        + RePaint-style latent blending using the 64×64 mask.
                safe { cb.onStage(requestId, "diffusing") }
                val fullPrompt = buildPrompt(prompt)
                val erasedRgbFp32 = bitmapToRgb01Chw(erased512)
                // Latent-space mask (64×64). 1 = regenerate, 0 = preserve, with
                // soft transition matching the feathered 512×512 mask. Use a
                // sharper sigma than the composite mask — the composite already
                // hides residual halo via [compositeAlphaSimple].
                val featheredForLatent = featherMask(mask512, sigma = 2.0f)
                val mask64 = downsampleMaskMean(featheredForLatent, 64)
                val genPng = NativeImageGen.nativeRunDiffusionImg2ImgPng(
                    bundleDir.absolutePath,
                    fullPrompt,
                    erasedRgbFp32,
                    mask64,
                    IMG2IMG_STRENGTH,
                    iterations,
                    seed,
                )
                if (genPng == null || genPng.isEmpty()) {
                    safe {
                        cb.onError(requestId, "GENERATE_FAILED",
                            "img2img returned empty — see logcat 'diffusion' tag")
                    }
                    return@launch
                }
                val gen512 = BitmapFactory.decodeByteArray(genPng, 0, genPng.size)
                    ?: run {
                        safe { cb.onError(requestId, "GENERATE_FAILED",
                            "img2img produced ${genPng.size} bytes but couldn't decode") }
                        return@launch
                    }
                Log.i(TAG, "img2img produced ${genPng.size} bytes → ${gen512.width}×${gen512.height}")

                // ---- 5. Hard-stamp original outside the mask so face/hair are bit-exact ----
                safe { cb.onStage(requestId, "decoding") }
                val feathered = featherMask(mask512, sigma = 4.0f)
                val composited = compositeAlphaSimple(photo512, gen512, feathered)
                photo512.recycle(); erased512.recycle(); gen512.recycle()

                val outPng = ByteArrayOutputStream().use { os ->
                    composited.compress(Bitmap.CompressFormat.PNG, 100, os)
                    os.toByteArray()
                }
                composited.recycle()

                // ---- 6. Deliver via PFD pipe ----
                val pipe = ParcelFileDescriptor.createPipe()
                val readEnd = pipe[0]
                val writeEnd = pipe[1]
                val writer = launch(Dispatchers.IO) {
                    try {
                        ParcelFileDescriptor.AutoCloseOutputStream(writeEnd).use { os ->
                            os.write(outPng)
                        }
                    } catch (t: Throwable) {
                        Log.w(TAG, "pfd writer failed rid=$requestId", t)
                    }
                }
                try {
                    safe { cb.onResult(requestId, readEnd, 512, 512) }
                } finally {
                    runCatching { readEnd.close() }
                    writer.join()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "outfit-swap failed rid=$requestId", t)
                safe { cb.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
            } finally {
                jobs.remove(requestId)
            }
        }
        jobs[requestId] = job
        return requestId
    }

    fun cancel(requestId: String) {
        jobs.remove(requestId)?.cancel()
    }

    fun close() {
        scope.cancel()
    }

    // ----------------------- helpers -----------------------

    private fun decodeAndResize512(pngBytes: ByteArray): Bitmap? {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val raw = BitmapFactory.decodeByteArray(pngBytes, 0, pngBytes.size, opts) ?: return null
        val side = min(raw.width, raw.height)
        val cx = (raw.width - side) / 2
        val cy = (raw.height - side) / 2
        val square = if (raw.width == side && raw.height == side) raw
        else Bitmap.createBitmap(raw, cx, cy, side, side)
        val resized = if (square.width == SIZE && square.height == SIZE) square
        else Bitmap.createScaledBitmap(square, SIZE, SIZE, /*filter=*/true)
        if (raw !== square && square !== resized) square.recycle()
        if (raw !== resized) raw.recycle()
        return resized
    }

    /**
     * Convert SegFormer's argmax classmap (segH × segW uint8) into a 512×512
     * binary mask using MI-GAN's convention: `0` = pixel to be erased,
     * `255` = pixel to keep.
     */
    private fun classmapToMask512(
        classmap: ByteArray, segH: Int, segW: Int, selectedClasses: Int,
    ): ByteArray {
        val out = ByteArray(SIZE * SIZE)
        for (y in 0 until SIZE) {
            val sy = (y * segH / SIZE).coerceAtMost(segH - 1)
            for (x in 0 until SIZE) {
                val sx = (x * segW / SIZE).coerceAtMost(segW - 1)
                val cls = classmap[sy * segW + sx].toInt() and 0xFF
                val isMasked = cls < 32 && (selectedClasses and (1 shl cls)) != 0
                out[y * SIZE + x] = if (isMasked) 0.toByte() else 0xFF.toByte()
            }
        }
        return out
    }

    /**
     * Separable Gaussian feather of the binary mask into a float alpha map in
     * `[0,1]`. Output convention: `1.0` = pixel was masked (use generated),
     * `0.0` = pixel was not masked (use erased), with soft transition at the
     * boundary. Truncates the kernel at ±3σ.
     */
    private fun featherMask(mask01: ByteArray, sigma: Float): FloatArray {
        // First convert to "1 where masked, 0 elsewhere".
        val src = FloatArray(SIZE * SIZE)
        for (i in src.indices) {
            src[i] = if ((mask01[i].toInt() and 0xFF) == 0) 1.0f else 0.0f
        }
        if (sigma <= 0f) return src

        val r = max(1, ceil(3.0f * sigma).toInt())
        val k = FloatArray(2 * r + 1)
        var ksum = 0.0f
        for (i in -r..r) {
            val v = exp(-(i * i) / (2.0f * sigma * sigma))
            k[i + r] = v
            ksum += v
        }
        for (i in k.indices) k[i] /= ksum

        val tmp = FloatArray(SIZE * SIZE)
        // Row pass.
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                var acc = 0.0f
                for (j in -r..r) {
                    val xx = (x + j).coerceIn(0, SIZE - 1)
                    acc += src[y * SIZE + xx] * k[j + r]
                }
                tmp[y * SIZE + x] = acc
            }
        }
        // Column pass.
        val out = FloatArray(SIZE * SIZE)
        for (x in 0 until SIZE) {
            for (y in 0 until SIZE) {
                var acc = 0.0f
                for (j in -r..r) {
                    val yy = (y + j).coerceIn(0, SIZE - 1)
                    acc += tmp[yy * SIZE + x] * k[j + r]
                }
                out[y * SIZE + x] = acc.coerceIn(0f, 1f)
            }
        }
        return out
    }

    /**
     * Straight feathered alpha blend between [outside] (mask=0) and [gen]
     * (mask=1). Used by the img2img path — the generated image already
     * contains body shading and perspective, so we just composite without
     * any luminance transfer (which would brighten gen pixels to match
     * the skin-filled erased region and wash the new garment color out).
     */
    private fun compositeAlphaSimple(
        outside: Bitmap, gen: Bitmap, alpha: FloatArray,
    ): Bitmap {
        val gen512 = if (gen.width == SIZE && gen.height == SIZE) gen
        else Bitmap.createScaledBitmap(gen, SIZE, SIZE, /*filter=*/true)

        val a = IntArray(SIZE * SIZE)
        val b = IntArray(SIZE * SIZE)
        outside.getPixels(a, 0, SIZE, 0, 0, SIZE, SIZE)
        gen512.getPixels(b, 0, SIZE, 0, 0, SIZE, SIZE)

        val out = IntArray(SIZE * SIZE)
        for (i in out.indices) {
            val w = alpha[i]
            if (w <= 0.001f) { out[i] = a[i]; continue }
            if (w >= 0.999f) { out[i] = b[i]; continue }
            val ar = (a[i] shr 16) and 0xFF
            val ag = (a[i] shr 8)  and 0xFF
            val ab =  a[i]         and 0xFF
            val br = (b[i] shr 16) and 0xFF
            val bg = (b[i] shr 8)  and 0xFF
            val bb =  b[i]         and 0xFF
            val r  = ((1 - w) * ar + w * br).toInt().coerceIn(0, 255)
            val g  = ((1 - w) * ag + w * bg).toInt().coerceIn(0, 255)
            val bc = ((1 - w) * ab + w * bb).toInt().coerceIn(0, 255)
            out[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or bc
        }
        if (gen512 !== gen) gen512.recycle()
        return Bitmap.createBitmap(out, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    /**
     * Mean-pool a 512×512 fp32 mask down to [outSize]×[outSize]. Used to
     * produce the 64×64 latent-space mask for RePaint latent blending — the
     * value at each output cell is the average of an 8×8 block of source
     * pixels (preserving the soft feather as graceful gradients).
     */
    private fun downsampleMaskMean(src: FloatArray, outSize: Int): FloatArray {
        val factor = SIZE / outSize
        require(factor * outSize == SIZE) { "outSize must divide $SIZE" }
        val out = FloatArray(outSize * outSize)
        val norm = 1.0f / (factor * factor)
        for (oy in 0 until outSize) {
            for (ox in 0 until outSize) {
                var acc = 0f
                val srcY0 = oy * factor
                val srcX0 = ox * factor
                for (iy in 0 until factor) {
                    val row = (srcY0 + iy) * SIZE + srcX0
                    for (ix in 0 until factor) {
                        acc += src[row + ix]
                    }
                }
                out[oy * outSize + ox] = acc * norm
            }
        }
        return out
    }

    private fun bitmapToRgb01Chw(bm: Bitmap): FloatArray {
        val pixels = IntArray(SIZE * SIZE)
        bm.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val out   = FloatArray(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            out[i              ] = ((p shr 16) and 0xFF) / 255f
            out[i +     plane  ] = ((p shr 8)  and 0xFF) / 255f
            out[i + 2 * plane  ] = ( p         and 0xFF) / 255f
        }
        return out
    }

    /**
     * Legacy luminance-preserving composite kept for the (no-longer-default)
     * txt2img path: re-illuminate each generated pixel with the original's
     * per-pixel luminance so a flat texture inherits body shading. The active
     * img2img path uses [compositeAlphaSimple] instead — its output is already
     * body-aware so no luminance trick is needed.
     */
    @Suppress("unused")
    private fun compositeBitmaps(
        erased: Bitmap, gen: Bitmap, alpha: FloatArray,
    ): Bitmap {
        val gen512 = if (gen.width == SIZE && gen.height == SIZE) gen
        else Bitmap.createScaledBitmap(gen, SIZE, SIZE, /*filter=*/true)

        val a = IntArray(SIZE * SIZE)
        val b = IntArray(SIZE * SIZE)
        erased.getPixels(a, 0, SIZE, 0, 0, SIZE, SIZE)
        gen512.getPixels(b, 0, SIZE, 0, 0, SIZE, SIZE)

        val out = IntArray(SIZE * SIZE)
        for (i in out.indices) {
            val w = alpha[i]
            val ar = (a[i] shr 16) and 0xFF
            val ag = (a[i] shr 8)  and 0xFF
            val ab =  a[i]         and 0xFF

            if (w <= 0.001f) {
                out[i] = a[i]
                continue
            }

            val br = (b[i] shr 16) and 0xFF
            val bg = (b[i] shr 8)  and 0xFF
            val bb =  b[i]         and 0xFF

            // Rec. 601 luminance — perceptually meaningful gray for body shading.
            val erLum = 0.299f * ar + 0.587f * ag + 0.114f * ab
            val genLum = (0.299f * br + 0.587f * bg + 0.114f * bb).coerceAtLeast(1f)

            // Re-illuminate gen so it inherits the original body's lighting.
            // Cap the scaling factor — extreme highlights/shadows would otherwise
            // wash the texture to pure white/black and lose hue.
            val scale = (erLum / genLum).coerceIn(0.3f, 2.2f)
            val newR = (br * scale).toInt().coerceIn(0, 255)
            val newG = (bg * scale).toInt().coerceIn(0, 255)
            val newB = (bb * scale).toInt().coerceIn(0, 255)

            val r = ((1 - w) * ar + w * newR).toInt().coerceIn(0, 255)
            val g = ((1 - w) * ag + w * newG).toInt().coerceIn(0, 255)
            val bComp = ((1 - w) * ab + w * newB).toInt().coerceIn(0, 255)
            out[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or bComp
        }
        if (gen512 !== gen) gen512.recycle()
        return Bitmap.createBitmap(out, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    /**
     * Wrap the user's outfit description in a person-wearing template. img2img
     * starts from the MI-GAN-erased body latent, so the network already knows
     * where the body is — we WANT it to paint the garment onto that body
     * rather than render a flat fabric swatch. The "same pose, same face"
     * suffix biases the model against drifting away from the original
     * portrait at the strengths we use (~0.7).
     */
    private fun buildPrompt(userPrompt: String): String {
        val cleaned = userPrompt.trim().trimEnd('.')
        if (cleaned.isEmpty()) return "portrait photograph, photorealistic"
        return "portrait of a person wearing $cleaned, photorealistic, " +
            "sharp focus, studio lighting, same pose, same face"
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (e: RemoteException) {
            Log.w(TAG, "callback remote dead", e)
        } catch (t: Throwable) {
            Log.w(TAG, "callback threw", t)
        }
    }

    companion object {
        const val OUTFIT_SWAP_DIR_NAME = "outfit-swap"  // legacy name; kept for callers
        private const val SIZE = 512
        // Below this pixel count we treat the mask as "garment not present"
        // and bail with NO_GARMENT_DETECTED rather than running a wasted
        // 10-second txt2img.
        private const val MIN_MASK_PIXELS = 256
        // img2img strength: fraction of the SD 1.5 schedule applied as noise to
        // the VAE-encoded input. 0.7 lets the garment paint over the body while
        // keeping pose + face stable; raise toward 1.0 for more prompt adherence.
        private const val IMG2IMG_STRENGTH = 0.7f

        // Bitfield over SegFormer's 18 ATR classes. Mirror of mask_ops.hpp.
        private const val UPPER_CLOTHES = 1 shl 4
        private const val SKIRT         = 1 shl 5
        private const val PANTS         = 1 shl 6
        private const val DRESS         = 1 shl 7
        private const val HAT           = 1 shl 1
        private const val BELT          = 1 shl 8
        private const val SCARF         = 1 shl 17

        @JvmStatic
        fun parseGarmentSpec(spec: String): Int? {
            val s = spec.trim().lowercase()
            if (s == "auto") {
                return UPPER_CLOTHES or SKIRT or PANTS or DRESS or HAT or BELT or SCARF
            }
            var bits = 0
            for (token in s.split(",").map { it.trim() }) {
                bits = bits or when (token) {
                    "upper-clothes" -> UPPER_CLOTHES
                    "skirt"         -> SKIRT
                    "pants"         -> PANTS
                    "dress"         -> DRESS
                    "hat"           -> HAT
                    "belt"          -> BELT
                    "scarf"         -> SCARF
                    else            -> return null
                }
            }
            return if (bits == 0) null else bits
        }
    }
}
