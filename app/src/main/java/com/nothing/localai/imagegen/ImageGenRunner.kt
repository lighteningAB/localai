package com.nothing.localai.imagegen

import android.content.Context
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
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "ImageGenRunner"

/**
 * Default install location for the SD model bundle on device. Layout matches an
 * extracted xororz HF bundle (e.g. `AbsoluteReality_qnn2.28_8gen2.zip` minus its
 * `output_512/qnn_models_8gen2/` wrapper):
 *
 *   filesDir/models/sd-v15-xororz/
 *     ├── tokenizer.json
 *     ├── clip_v2.mnn          (text encoder — MNN, not QNN)
 *     ├── pos_emb.bin
 *     ├── token_emb.bin
 *     ├── vae_encoder.bin      (QNN context binary)
 *     ├── vae_decoder.bin      (QNN context binary)
 *     ├── unet.bin             (QNN context binary, the heavy file)
 *     └── *.patch              (optional resolution-specific UNet overlays)
 *
 * Push with `scripts/push-diffusion-bundle.sh <extracted-dir>` after running
 * `unzip <bundle>.zip` on the host. The push script's default DEST_NAME is kept
 * in sync with this constant.
 */
const val DEFAULT_DIFFUSION_DIR_NAME = "sd-v15-xororz"

/**
 * On-device image generator. **Status: stub** through Phase 7; Phase 8 wires this to
 * native via [NativeImageGen].
 *
 * Plan B (active): clean-room native C++ + Qualcomm QNN SDK consuming xororz HF
 * pre-converted SD 1.5 bundles. See `IMAGE-GEN-PLAN.md` for the 8-phase implementation
 * plan and `KNOWN-ISSUES.md` for why MediaPipe (Plan A) and LiteRT/AI Hub (Plan A2)
 * were abandoned.
 *
 * What's already wired:
 *   - AIDL surface: `ILocalAiService.generateImage`/`cancelImageGen`
 *   - Service: `LocalAiService.kt` — foreground-notification lifecycle
 *   - Bridge: `Aiwidget/.../LocalAiBridge.kt` — emits `LocalAi.image` events
 *   - Widget: `Aiwidget/testwidgets/image-gen-1/src/App.tsx`
 *
 * What's stubbed here: [generate] returns `BUNDLE_MISSING` or `NOT_IMPLEMENTED`.
 * Phase 8 replaces the `NOT_IMPLEMENTED` branch with a `NativeImageGen.nativeGenerate`
 * call that drives the diffusion loop in `libimagegen.so`.
 */
class ImageGenRunner(
    private val ctx: Context,
    private val diffusionDirName: String = DEFAULT_DIFFUSION_DIR_NAME,
) {

    private val jobs = ConcurrentHashMap<String, Job>()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val modelDir: File
        get() = File(File(ctx.filesDir, "models"), diffusionDirName)

    fun isReady(): Boolean {
        val d = modelDir
        if (!d.isDirectory) return false
        // Required artifacts of an extracted xororz SD-QNN bundle. The native
        // bundle_loader.cpp performs the authoritative check; this is the cheap
        // pre-flight that the service uses before deciding to spawn native init.
        return listOf(
            "tokenizer.json",
            "clip_v2.mnn",
            "pos_emb.bin",
            "token_emb.bin",
            "vae_encoder.bin",
            "vae_decoder.bin",
            "unet.bin",
        ).all { File(d, it).exists() }
    }

    fun generate(prompt: String, iterations: Int, seed: Long, cb: IImageGenCallback): String {
        val requestId = UUID.randomUUID().toString()
        val job = scope.launch {
            try {
                if (!isReady()) {
                    safe {
                        cb.onError(
                            requestId,
                            "BUNDLE_MISSING",
                            "SD bundle not present at ${modelDir.absolutePath}. " +
                                "Download from Qualcomm AI Hub and push with " +
                                "scripts/push-diffusion-bundle.sh."
                        )
                    }
                    return@launch
                }

                // Native call is synchronous and runs the full pipeline (text
                // encode + UNet loop + VAE decode + PNG encode). No mid-flight
                // step callbacks yet — Phase 8 will add a NativeCallback that
                // streams onStep events. For now the widget gets a single
                // onResult once the PNG is ready.
                Log.i(TAG, "generate rid=$requestId prompt='${prompt.take(40)}' iters=$iterations")
                val png = NativeImageGen.nativeRunDiffusionPng(
                    modelDir.absolutePath, prompt, iterations, seed,
                )
                if (png == null || png.isEmpty()) {
                    safe {
                        cb.onError(
                            requestId,
                            "GENERATE_FAILED",
                            "nativeRunDiffusionPng returned no bytes — see logcat " +
                                "tags 'diffusion' / 'imagegen' / 'qnn_session'."
                        )
                    }
                    return@launch
                }

                // Hand the PNG to the client through a one-shot pipe. The
                // write end is closed by AutoCloseOutputStream after we drain
                // the byte array; the binder hands the read end across the
                // process boundary to the widget side.
                val pipe = ParcelFileDescriptor.createPipe()
                val readEnd  = pipe[0]
                val writeEnd = pipe[1]
                val writer = launch(Dispatchers.IO) {
                    try {
                        ParcelFileDescriptor.AutoCloseOutputStream(writeEnd).use { os ->
                            os.write(png)
                        }
                    } catch (t: Throwable) {
                        Log.w(TAG, "pfd writer failed rid=$requestId", t)
                    }
                }
                try {
                    safe {
                        cb.onResult(requestId, readEnd, /*width=*/512, /*height=*/512)
                    }
                } finally {
                    // The PFD read end was duplicated across the binder; closing
                    // our local handle here doesn't tear down the client's copy.
                    runCatching { readEnd.close() }
                    writer.join()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "generate failed rid=$requestId", t)
                safe { cb.onError(requestId, "GENERATE_FAILED", t.message ?: "unknown") }
            } finally {
                jobs.remove(requestId)
            }
        }
        jobs[requestId] = job
        return requestId
    }

    fun cancel(requestId: String) {
        jobs[requestId]?.cancel()
    }

    fun close() {
        scope.cancel()
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (_: RemoteException) { /* client died */ }
    }
}
