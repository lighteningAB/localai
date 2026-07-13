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

    // ===== Active multimodal model selection =====
    // Switch which Gemma 4 .litertlm bundle backs generate()/addImage()/addAudio().
    // setActiveModel persists the choice and reloads the engine on the next
    // request (closing any live session). Returns the now-active model id (the
    // request id if accepted, the previous id if unknown/unchanged).
    // Used by StatusActivity's model switch; safe for hosts to call too.
    String getActiveModel();
    String setActiveModel(String modelId);

    // ===== Constrained generation (GBNF) =====
    // Like generate(), but constrains output to a GBNF grammar (constrained
    // decoding — e.g. force valid UI-spec JSON or a tool call). Only the
    // llama.cpp backend (Gemma 4 12B) enforces the grammar; LiteRT-LM models
    // have no grammar API and fall back to plain generation (grammar ignored),
    // so callers must keep their repair/validation net for those. Empty grammar
    // behaves exactly like generate(). Returns a requestId for cancel().
    String generateConstrained(String sessionId, String prompt, String grammar, ITokenCallback cb);
}
