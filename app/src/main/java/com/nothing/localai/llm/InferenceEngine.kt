package com.nothing.localai.llm

import android.os.ParcelFileDescriptor
import com.nothing.localai.ITokenCallback

/**
 * Backend-neutral seam so the service and [com.nothing.localai.session.SessionRegistry]
 * don't care whether a model runs on LiteRT-LM ([LlmRunner]/[ChatSession]) or
 * llama.cpp ([LlamaCppRunner]/[LlamaSession]). [EngineManager] routes by the
 * active model's [Backend].
 *
 * The two runtimes never run concurrently — a 12B GGUF and a Gemma-4 .litertlm
 * together would blow the RAM budget — so [EngineManager] tears one down before
 * bringing the other up.
 */
interface InferenceRunner {
    fun activeModelId(): String
    /** Swap active model within this runtime. Caller closes live sessions first. */
    fun setActiveModel(newId: String): Boolean
    fun newSession(sessionId: String): InferenceSession
    fun cancel(requestId: String)
    fun close()
}

/** One conversation / KV cache. Mirrors the ChatSession surface. */
interface InferenceSession {
    fun generate(requestId: String, prompt: String, cb: ITokenCallback)

    /**
     * Like [generate] but constrains output to a GBNF [grammar] (constrained
     * decoding). Only the llama.cpp backend enforces it; the default here
     * ignores the grammar and falls back to plain [generate] — LiteRT-LM has no
     * grammar API, so a LiteRT model degrades to best-effort (the caller keeps
     * its repair/loop-guard net). Empty grammar == [generate].
     */
    fun generateConstrained(requestId: String, prompt: String, grammar: String, cb: ITokenCallback) =
        generate(requestId, prompt, cb)

    fun addImage(jpegFd: ParcelFileDescriptor)
    fun addAudio(pcmFd: ParcelFileDescriptor, sampleRate: Int)
    fun reset()
    fun close()
}
