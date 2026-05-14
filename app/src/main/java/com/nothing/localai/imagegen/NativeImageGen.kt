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
}
