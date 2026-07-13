// JNI bridge for the llama.cpp inference engine (libllmcpp.so).
//
// Second LLM backend beside the Kotlin LiteRT-LM path, for models the
// LiteRT-LM GPU runtime cannot host — chiefly Gemma 4 12B, whose fp32 logits
// tensor overflows the Adreno 1 GB single-allocation cap on the GPU backend
// (see ModelId.kt / LiteRT-LM issue #2461). llama.cpp runs the 12B QAT Q4_0
// GGUF on the CPU, sidestepping that cap entirely.
//
// Two native handles mirror the LiteRT-LM split:
//   Model    (llama_model)   ~ Engine       — the loaded weights, shared.
//   Context  (llama_context) ~ Conversation — one KV cache / chat history.
// The Kotlin side keeps one Model live at a time (RAM) and one Context per
// active session (SessionRegistry single-active policy).

#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "llama_bridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::atomic<bool> g_backend_ready{false};

void ensure_backend() {
    bool expected = false;
    if (g_backend_ready.compare_exchange_strong(expected, true)) {
        // The fused RMS_NORM+MUL CPU kernel SIGSEGVs on this build during
        // multi-threaded large-batch prefill (symbolized: ops.cpp
        // ggml_compute_forward_rms_norm_f32<fused> on a secondary thread,
        // 512-token chunk). ggml checks this env at graph time — disable
        // fusion outright; the unfused path is marginally slower and stable.
        setenv("GGML_CPU_DISABLE_FUSION", "1", 1);
        llama_backend_init();
        LOGI("llama backend initialized (cpu fusion disabled)");
    }
}

// A loaded model (weights). Long-lived; one at a time on the Kotlin side.
struct ModelHandle {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
};

// One chat session: KV cache + sampler + cancel flag. Positions are tracked
// automatically by llama_decode, so successive generate() calls into the same
// context continue the conversation (multi-turn) until reset.
struct ContextHandle {
    llama_context* ctx = nullptr;
    int32_t seed = 42;
    std::atomic<bool> cancel{false};
    bool started = false;   // BOS + no leading turn-sep only on the first turn
};

// Build a fresh sampler chain for one generation. If grammar is non-empty it is
// applied FIRST (masks tokens the GBNF grammar forbids to -inf) so the
// downstream top_k/top_p/temp/dist only ever pick a grammar-valid token — this
// is the constrained-decoding guarantee LiteRT-LM can't provide. Returns null
// if the grammar fails to parse (caller falls back to unconstrained).
llama_sampler* build_sampler(const llama_vocab* vocab, int32_t seed, const char* grammar) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (grammar && grammar[0] != '\0') {
        llama_sampler* g = llama_sampler_init_grammar(vocab, grammar, "root");
        if (!g) {
            LOGE("grammar failed to parse; falling back to unconstrained");
            llama_sampler_free(chain);
            return nullptr;
        }
        llama_sampler_chain_add(chain, g);
    }
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t) seed));
    return chain;
}

inline ModelHandle* asModel(jlong p) { return reinterpret_cast<ModelHandle*>(p); }
inline ContextHandle* asCtx(jlong p) { return reinterpret_cast<ContextHandle*>(p); }

std::string piece_for(const llama_vocab* vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, /*special=*/false);
    if (n <= 0) return std::string();
    return std::string(buf, n);
}

} // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativePing(JNIEnv* env, jobject) {
    ensure_backend();
    const char* sysinfo = llama_print_system_info();
    std::string out = "llmcpp-native-v0 | ";
    out += (sysinfo ? sysinfo : "(no sysinfo)");
    LOGI("ping: %s", out.c_str());
    return env->NewStringUTF(out.c_str());
}

// Load GGUF weights. n_gpu_layers = 0 (CPU only for now). Returns 0 on failure.
JNIEXPORT jlong JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeLoadModel(
        JNIEnv* env, jobject, jstring jpath) {
    ensure_backend();
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // CPU: the whole point — no Adreno alloc cap
    llama_model* model = llama_model_load_from_file(path, mp);
    LOGI("load model: %s -> %p", path, (void*)model);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) return 0;
    auto* h = new ModelHandle{model, llama_model_get_vocab(model)};
    return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeFreeModel(JNIEnv*, jobject, jlong modelPtr) {
    auto* h = asModel(modelPtr);
    if (!h) return;
    if (h->model) llama_model_free(h->model);
    delete h;
}

// Create a chat context (KV cache) + sampler chain. Returns 0 on failure.
JNIEXPORT jlong JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeCreateContext(
        JNIEnv*, jobject, jlong modelPtr, jint nCtx, jint nThreads, jint seed) {
    auto* m = asModel(modelPtr);
    if (!m || !m->model) return 0;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) nCtx;
    // Chunk prefill at 512 rather than the full context. n_batch = n_ctx (4096)
    // allocated a huge transient compute buffer and processed the whole system
    // prompt in one CPU spike; 512 bounds peak memory and smooths the load.
    cp.n_batch = (uint32_t) (nCtx < 512 ? nCtx : 512);
    if (nThreads > 0) {
        // Asymmetric threading: decode is the SUSTAINED phase (minutes at 100%
        // — what overheated the board), so it gets the caller's low cap; prefill
        // is a short burst, so give it more cores to cut time-to-first-token on
        // the app's ~1300-token system prompt.
        cp.n_threads = nThreads;
        // 6 prefill threads peaked 103°C on a ~1300-token prompt (near trip);
        // 4 keeps the burst inside thermal budget at ~2/3 the speed.
        cp.n_threads_batch = nThreads * 2 > 4 ? 4 : nThreads * 2;
    }
    llama_context* ctx = llama_init_from_model(m->model, cp);
    if (!ctx) return 0;

    auto* h = new ContextHandle();
    h->ctx = ctx;
    h->seed = seed;
    LOGI("create context: nCtx=%d nThreads=%d -> %p", nCtx, nThreads, (void*)ctx);
    return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeFreeContext(JNIEnv*, jobject, jlong ctxPtr) {
    auto* h = asCtx(ctxPtr);
    if (!h) return;
    if (h->ctx) llama_free(h->ctx);
    delete h;
}

// Clear KV cache + reset chat state without reloading the model.
JNIEXPORT void JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeResetContext(JNIEnv*, jobject, jlong ctxPtr) {
    auto* h = asCtx(ctxPtr);
    if (!h || !h->ctx) return;
    llama_memory_clear(llama_get_memory(h->ctx), /*data=*/true);
    h->started = false;
    h->cancel.store(false);
}

// Signal an in-flight nativeGenerate to stop (checked between tokens).
JNIEXPORT void JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeCancel(JNIEnv*, jobject, jlong ctxPtr) {
    auto* h = asCtx(ctxPtr);
    if (h) h->cancel.store(true);
}

// Run one user turn. Wraps `prompt` in Gemma 4's chat template, decodes it into
// the persistent KV cache, then samples until <end_of_turn>/EOG or maxTokens.
// Each generated token is streamed to sink.onToken(String); the full text is
// also returned. Blocking — caller runs it on a background thread.
JNIEXPORT jstring JNICALL
Java_com_nothing_localai_llm_NativeLlama_nativeGenerate(
        JNIEnv* env, jobject, jlong ctxPtr, jlong modelPtr,
        jstring jprompt, jstring jgrammar, jint maxTokens, jobject sink) {
    auto* c = asCtx(ctxPtr);
    auto* m = asModel(modelPtr);
    if (!c || !c->ctx || !m || !m->model) return env->NewStringUTF("");

    c->cancel.store(false);

    // Per-call sampler chain, optionally GBNF-constrained. Empty grammar = free
    // sampling. A grammar that fails to parse falls back to unconstrained.
    const char* grammar = jgrammar ? env->GetStringUTFChars(jgrammar, nullptr) : nullptr;
    llama_sampler* smpl = build_sampler(m->vocab, c->seed, grammar);
    if (!smpl) smpl = build_sampler(m->vocab, c->seed, nullptr);
    if (grammar) env->ReleaseStringUTFChars(jgrammar, grammar);

    // Resolve sink.onToken(String) once for the token stream.
    jmethodID onToken = nullptr;
    if (sink) {
        jclass sinkCls = env->GetObjectClass(sink);
        onToken = env->GetMethodID(sinkCls, "onToken", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(sinkCls);
    }

    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    // Gemma 4 chat turn. NOTE: Gemma 4 does NOT use gemma3's <start_of_turn>;
    // its format (from the model's own chat_template) is asymmetric turn/channel
    // markers: a turn is  <|turn>ROLE\n ... <turn|>\n , and the generation prompt
    // opens the model turn plus an empty thought channel (thinking disabled) so
    // the model goes straight to its answer:
    //   <|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    // parse_special=true tokenizes the <|...> markers as specials; BOS is added
    // by the tokenizer on the first turn of a fresh context only.
    std::string templated = "<|turn>user\n";
    templated += prompt;
    templated += "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    env->ReleaseStringUTFChars(jprompt, prompt);

    const bool add_bos = !c->started;
    int32_t need = -llama_tokenize(m->vocab, templated.c_str(), (int32_t) templated.size(),
                                   nullptr, 0, add_bos, /*parse_special=*/true);
    std::vector<llama_token> toks(need > 0 ? need : 0);
    int32_t nTok = llama_tokenize(m->vocab, templated.c_str(), (int32_t) templated.size(),
                                  toks.data(), (int32_t) toks.size(), add_bos, true);
    if (nTok < 0) { llama_sampler_free(smpl); return env->NewStringUTF(""); }
    toks.resize(nTok);

    // Prefill the prompt in n_batch-sized chunks. Submitting the whole prompt
    // as one batch aborts the process when it exceeds n_batch (the app's routed
    // system prompt is ~1300 tokens vs n_batch 512 — this killed :inference).
    LOGI("prefill: %d tokens (add_bos=%d)", (int) toks.size(), (int) add_bos);
    const int32_t nBatch = (int32_t) llama_n_batch(c->ctx);
    for (int32_t off = 0; off < (int32_t) toks.size(); off += nBatch) {
        const int32_t n = (int32_t) toks.size() - off < nBatch
            ? (int32_t) toks.size() - off : nBatch;
        if (llama_decode(c->ctx, llama_batch_get_one(toks.data() + off, n)) != 0) {
            LOGE("prefill decode failed at offset %d", off);
            llama_sampler_free(smpl);
            return env->NewStringUTF("");
        }
    }
    c->started = true;
    LOGI("prefill decoded");

    std::string full;
    const int cap = maxTokens > 0 ? maxTokens : 2048;
    for (int i = 0; i < cap; i++) {
        if (c->cancel.load()) { LOGI("generate cancelled at %d tokens", i); break; }
        llama_token id = llama_sampler_sample(smpl, c->ctx, -1);
        if (i == 0) LOGI("first token sampled");
        else if ((i % 16) == 0) LOGI("... %d tokens", i);
        if (llama_vocab_is_eog(m->vocab, id)) { LOGI("EOG at token %d", i); break; }

        std::string piece = piece_for(m->vocab, id);
        if (!piece.empty()) {
            full += piece;
            if (onToken) {
                jstring js = env->NewStringUTF(piece.c_str());
                env->CallVoidMethod(sink, onToken, js);
                env->DeleteLocalRef(js);
                if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
            }
        }
        if (llama_decode(c->ctx, llama_batch_get_one(&id, 1)) != 0) {
            LOGE("decode failed at token %d", i);
            break;
        }
    }
    llama_sampler_free(smpl);
    return env->NewStringUTF(full.c_str());
}

} // extern "C"
