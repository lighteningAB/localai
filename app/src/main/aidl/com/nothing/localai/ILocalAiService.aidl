// ILocalAiService.aidl
// Contract between Aiwidget (host) and com.nothing.localai (model service).
// Mirrored byte-identical in the Aiwidget project. Methods may only be APPENDED
// at the end; reordering or removing breaks binary compatibility.
package com.nothing.localai;

import com.nothing.localai.ITokenCallback;
import com.nothing.localai.IModelStatusCallback;
import com.nothing.localai.IImageGenCallback;
import android.os.ParcelFileDescriptor;

interface ILocalAiService {

    // ===== Service =====
    int getApiVersion();

    // ===== Model lifecycle =====
    String getModelStatus(String modelId);
    void ensureModel(String modelId, IModelStatusCallback cb);

    // ===== LLM =====
    void createSession(String sessionId);
    void releaseSession(String sessionId);
    void resetSession(String sessionId);
    String generate(String sessionId, String prompt, ITokenCallback cb);
    void cancel(String requestId);

    // ===== Vision =====
    String classifyImage(in ParcelFileDescriptor jpegFd, int topK);

    // ===== Audio =====
    String transcribe(in ParcelFileDescriptor pcmFd, int sampleRate);
    void speak(String text);

    // ===== Multimodal LLM input (Gemma 3n) =====
    void addImage(String sessionId, in ParcelFileDescriptor jpegFd);
    void addAudio(String sessionId, in ParcelFileDescriptor pcmFd, int sampleRate);

    // ===== Image generation (MediaPipe Image Generator) =====
    // Diffusion runs against a separate model directory (MobileDiffusion or quantized SD v1.5
    // converted via image_generator_converter). Independent of the LLM session pool: no KV
    // cache to preserve. Returns a requestId for cancel().
    String generateImage(String prompt, int iterations, long seed, IImageGenCallback cb);
    void cancelImageGen(String requestId);

    // ===== Outfit swap (SD 1.5 inpaint + SegFormer-B2-Clothes) =====
    // Identity-preserving text-driven outfit edit. The service segments the input photo,
    // builds an inpaint mask from `garmentSpec`, and runs SD 1.5 inpaint UNet to repaint
    // the masked region. Single-subject portraits only in v1.
    //
    // `garmentSpec` accepts:
    //   "upper-clothes" | "skirt,pants" | "dress" | "upper-clothes,skirt,pants,dress" | "auto"
    // Unknown values fail-fast via IImageGenCallback.onError(code="BAD_GARMENT_SPEC").
    //
    // Stages stream via IImageGenCallback.onStage; the final PNG arrives via onResult.
    // See PLAN-OUTFIT-SWAP.md.
    String generateOutfitSwap(
        in ParcelFileDescriptor inputPng,
        String prompt,
        String garmentSpec,
        int iterations,
        long seed,
        IImageGenCallback cb);

    void cancelOutfitSwap(String requestId);

    // ===== SDXL image generation (apiVersion >= 2) =====
    // Independent SDXL-Lightning pipeline running its own QNN UNet + dual
    // MNN text encoders (CLIP-L + OpenCLIP-G) on V73. Routed by the service
    // to ImageGenXLRunner; does not share session or memory with generateImage
    // (which targets SD 1.5). Distinct method so consumers can pick at the
    // call site rather than via opaque model-id routing.
    //
    // Callback semantics identical to generateImage: onStep streams progress,
    // onResult delivers a 1024×1024 PNG via PFD, onError surfaces failure
    // codes. Returns a requestId for cancelImageGenXL().
    //
    // Callers MUST verify getApiVersion() >= 2 before invoking; older
    // services will throw TransactionTooLargeException / DeadObjectException
    // on the binder call.
    String generateImageXL(String prompt, int iterations, long seed, IImageGenCallback cb);

    void cancelImageGenXL(String requestId);
}
