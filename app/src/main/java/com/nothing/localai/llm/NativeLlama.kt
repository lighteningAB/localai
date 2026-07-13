package com.nothing.localai.llm

import android.util.Log

/**
 * JNI surface for the llama.cpp engine (libllmcpp.so). Second LLM backend
 * beside the Kotlin LiteRT-LM path — see [LlamaCppEngine] for the high-level
 * wrapper. Kept as a thin object so the native methods have a stable, minimal
 * binding point.
 *
 * The 12B QAT Q4_0 GGUF runs here on the CPU because the LiteRT-LM GPU runtime
 * cannot host it (Adreno 1 GB allocation cap on the fp32 logits tensor).
 *
 * Two native handles, mirroring the LiteRT-LM split:
 *  - **model** (`llama_model`) ~ Engine: loaded weights, one live at a time.
 *  - **context** (`llama_context`) ~ Conversation: one KV cache / chat history.
 *
 * All pointers are opaque `Long`s; the caller owns their lifetime and must call
 * the matching `free`. Generation is blocking — run it off the main thread.
 */
object NativeLlama {

    /** Receives streamed tokens during [nativeGenerate]. Called on the caller's
     *  thread (the coroutine's IO thread), one call per decoded token. */
    interface TokenSink {
        fun onToken(text: String)
    }

    @Volatile private var loaded = false

    /** Loads libllmcpp.so once. Safe to call repeatedly. Returns false if the
     *  library is absent or failed to link (device without the native lib). */
    @Synchronized
    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return try {
            System.loadLibrary("llmcpp")
            loaded = true
            true
        } catch (t: Throwable) {
            Log.e("NativeLlama", "failed to load libllmcpp.so", t)
            false
        }
    }

    /** Confirms the native lib links llama.cpp and reports ggml CPU features. */
    external fun nativePing(): String

    /** Load GGUF weights (CPU). Returns an opaque model pointer, or 0 on failure. */
    external fun nativeLoadModel(path: String): Long
    external fun nativeFreeModel(modelPtr: Long)

    /** Create a chat context (KV cache) + sampler over a loaded model.
     *  Returns an opaque context pointer, or 0 on failure. */
    external fun nativeCreateContext(modelPtr: Long, nCtx: Int, nThreads: Int, seed: Int): Long
    external fun nativeFreeContext(ctxPtr: Long)

    /** Clear KV cache + chat state without reloading the model. */
    external fun nativeResetContext(ctxPtr: Long)

    /** Signal an in-flight [nativeGenerate] to stop (checked between tokens). */
    external fun nativeCancel(ctxPtr: Long)

    /** Run one user turn: templated + decoded into the KV cache, sampled until
     *  end-of-turn or [maxTokens]. Streams tokens to [sink]; returns full text.
     *  [grammar] is a GBNF grammar string; "" = unconstrained. A non-empty
     *  grammar guarantees every sampled token is grammar-valid (constrained
     *  decoding) — the capability LiteRT-LM lacks. */
    external fun nativeGenerate(
        ctxPtr: Long,
        modelPtr: Long,
        prompt: String,
        grammar: String,
        maxTokens: Int,
        sink: TokenSink,
    ): String
}
