package com.nothing.localai.imagegen

/**
 * JNI surface for the native image-generation engine (Plan B — QNN-direct in-tree).
 *
 * Phase 1 (current): only [nativePing] is wired, used to prove the toolchain works.
 * Later phases will add nativeInit / nativeGenerate / nativeCancel / nativeClose plus
 * a NativeCallback interface for step/result/error callbacks. See `IMAGE-GEN-PLAN.md`
 * §3.8 for the eventual surface.
 */
object NativeImageGen {

    init {
        System.loadLibrary("imagegen")
    }

    /** Returns a hardcoded build-tag string from `imagegen.cpp`. Used to verify JNI load. */
    external fun nativePing(): String

    /**
     * Set `ADSP_LIBRARY_PATH` in the native process environment. Must be
     * called BEFORE the first QNN context-from-binary call: FastRPC uses this
     * path to load the Hexagon Skel onto the DSP. Pass
     * `applicationInfo.nativeLibraryDir` so the per-Hexagon-revision Skels
     * packaged in lib/arm64-v8a/ are discoverable.
     */
    external fun nativeSetAdspLibraryPath(path: String): Boolean

    /**
     * Phase 4a verification: open a QNN context binary at [path] (e.g.
     * `<filesDir>/models/sd-v15-xororz/unet.bin`), use QnnSystemContext to
     * inspect its metadata, return a multi-line report (graph names + tensor
     * shapes). A string starting with `ERROR:` indicates failure — caller
     * should log it. No execution happens here; this is metadata-only.
     */
    external fun nativeInspectQnnBinary(path: String): String

    /**
     * Phase 4b verification (metadata): load `clip_v2.mnn`, return a report of
     * the model's input/output shapes + active backend (OpenCL or CPU).
     */
    external fun nativeInspectMnnModel(path: String): String

    /**
     * Phase 4b verification (forward): expects the extracted xororz bundle at
     * [bundleDir] (containing `tokenizer.json`, `token_emb.bin`, `pos_emb.bin`,
     * `clip_v2.mnn`). Tokenizes [prompt], builds the [1,77,768] fp32 input
     * embedding via the bundle's lookup tables, runs `clip_v2.mnn` forward,
     * returns the `last_hidden_state` flat (size 1*77*768 = 59136). Returns
     * null on error (see logcat tag `imagegen` for the reason).
     */
    external fun nativeRunMnnTextEncode(bundleDir: String, prompt: String): FloatArray?

    /**
     * Phase 6 entry point: drive the full SD 1.5 diffusion loop with
     * classifier-free guidance using QNN UNet + MNN CLIP + DPM-Solver++.
     *
     * Returns the final latent tensor flat (fp32, length 1*4*64*64 = 16384) on
     * success, or an empty FloatArray on failure (multi-line diagnostic is in
     * logcat under tag `diffusion` / `imagegen`).
     *
     * Phase 7 will pipe this into the VAE decoder + PNG encode; for Phase 6
     * verification, finite latents + step 1/N…N/N logcat is the bar.
     */
    external fun nativeRunDiffusion(
        bundleDir: String,
        prompt: String,
        iters: Int,
        seed: Long,
    ): FloatArray?

    /**
     * Phase 7 entry point: drive the full SD 1.5 pipeline — text encode + UNet
     * diffusion loop (CFG) + VAE decode + PNG encode — and return the encoded
     * PNG bytes. Returns null on failure (multi-line diagnostic in logcat tags
     * `imagegen` / `diffusion`).
     *
     * This is the synchronous one-shot used by both the boot probe (writes
     * `filesDir/sd-debug.png`) and [ImageGenRunner] (pipes through a
     * ParcelFileDescriptor to the widget).
     */
    external fun nativeRunDiffusionPng(
        bundleDir: String,
        prompt: String,
        iters: Int,
        seed: Long,
    ): ByteArray?

    /**
     * Phase 6 img2img entry — see PLAN-IMG2IMG.md. Given an RGB image as
     * [inputRgbFp32] (flat 3×512×512 CHW [0,1]) and a [strength] in [0,1],
     * runs SD 1.5 with the latent seeded from the VAE-encoded input at
     * `t = strength·999` instead of from pure noise, then the schedule tail,
     * VAE decode, and PNG encode. Returns the PNG bytes on success or null
     * on failure (diagnostic in logcat tags `imagegen` / `diffusion`).
     *
     * Used by [OutfitSwapRunner] to seed the SD UNet from the MI-GAN-erased
     * portrait so generated garment texture inherits body shape, perspective,
     * and lighting.
     */
    external fun nativeRunDiffusionImg2ImgPng(
        bundleDir: String,
        prompt: String,
        inputRgbFp32: FloatArray,
        mask64Fp32: FloatArray,
        strength: Float,
        iters: Int,
        seed: Long,
    ): ByteArray?

    /**
     * Phase 0c+d entry point (outfit-swap, see [PLAN-OUTFIT-SWAP.md]): run
     * `mattmdjaga/segformer_b2_clothes` on a 512×512 RGB image and return a
     * colored visualization of the argmaxed class map as PNG bytes.
     *
     * - [modelBinPath]: absolute path to `segformer_b2_clothes.bin` (the AI Hub
     *   compile output; push to device via adb for Phase 0 testing).
     * - [inputRgbFp32]: flat 3×512×512 fp32 buffer, CHW row-major, normalized
     *   to ImageNet stats (`(rgb01 - mean) / std`, mean=[.485,.456,.406],
     *   std=[.229,.224,.225]). Caller does the resize + normalize on the JVM
     *   side via Bitmap.
     *
     * Returns null on failure — full diagnostic in logcat tag `segformer`.
     */
    external fun nativeRunSegformerMaskPng(
        modelBinPath: String,
        inputRgbFp32: FloatArray,
    ): ByteArray?

    /**
     * Outfit-swap Phase 2/3 entry point. Runs the full pipeline on-device:
     * SegFormer-B2-Clothes → mask_ops → CLIP + VAE encode → SD 1.5 inpaint
     * UNet diffusion (DDIM, CFG) → VAE decode → PNG. Synchronous; returns
     * the encoded PNG bytes on success, null on failure. Multi-line
     * diagnostic in logcat tag `outfit_swap` (and the per-stage tags
     * `segformer` / `diffusion`).
     *
     * - [rawRgbFp32] flat 3×512×512 fp32, CHW row-major, values in [0,1]
     *   (no normalization — the pipeline applies ImageNet + VAE norms internally).
     * - [bundleDir] absolute path to the xororz SD-QNN bundle dir
     *   (provides CLIP MNN + VAE encoder/decoder + tokenizer).
     * - [segformerBinPath] absolute path to the AI Hub-compiled SegFormer .bin.
     * - [inpaintUnetBinPath] absolute path to the AI Hub-compiled SD 1.5
     *   inpaint UNet .bin (9-channel input).
     * - [selectedClasses] bitfield over SegFormer's 18 ATR classes. See
     *   mask_ops.hpp `kGarment*` presets; use `kGarmentTop` (bit 4) for
     *   upper-clothes, etc.
     */
    external fun nativeRunOutfitSwap(
        rawRgbFp32: FloatArray,
        bundleDir: String,
        segformerBinPath: String,
        inpaintUnetBinPath: String,
        prompt: String,
        selectedClasses: Int,
        iterations: Int,
        seed: Long,
    ): ByteArray?
}
