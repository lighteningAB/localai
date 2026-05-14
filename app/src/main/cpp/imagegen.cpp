#include <jni.h>
#include <android/log.h>

#include <cstdlib>

#include <sstream>
#include <string>
#include <vector>

#include "diffusion.hpp"
#include "mnn_session.hpp"
#include "qnn_session.hpp"
#include "tokenizer.hpp"

namespace {
constexpr const char* kTag = "imagegen";
constexpr const char* kPingResponse = "imagegen-native-v0";

std::string jstringToStdString(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* p = env->GetStringUTFChars(s, nullptr);
    std::string out = p ? p : "";
    if (p) env->ReleaseStringUTFChars(s, p);
    return out;
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativePing(JNIEnv* env, jobject /*thiz*/) {
    __android_log_print(ANDROID_LOG_INFO, kTag, "nativePing() invoked");
    return env->NewStringUTF(kPingResponse);
}

// Point FastRPC at the directory containing the per-Hexagon-rev Skel libs
// (e.g. /data/app/.../lib/arm64/libQnnHtpV73Skel.so). Must be called BEFORE
// QnnContext_createFromBinary or the DSP cannot load the skel. Returns true
// on success.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeSetAdspLibraryPath(JNIEnv* env,
                                                                          jobject /*thiz*/,
                                                                          jstring path) {
    const std::string p = jstringToStdString(env, path);
    if (p.empty()) return JNI_FALSE;
    // Append the current value so we don't break any system paths that were
    // already configured (Qualcomm runtimes occasionally inject defaults).
    const char* existing = ::getenv("ADSP_LIBRARY_PATH");
    std::string combined = p;
    if (existing && *existing) {
        combined += ":";
        combined += existing;
    }
    if (::setenv("ADSP_LIBRARY_PATH", combined.c_str(), 1) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "setenv ADSP_LIBRARY_PATH failed");
        return JNI_FALSE;
    }
    __android_log_print(ANDROID_LOG_INFO, kTag,
                        "ADSP_LIBRARY_PATH=%s", combined.c_str());
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeInspectQnnBinary(JNIEnv* env,
                                                                        jobject /*thiz*/,
                                                                        jstring path) {
    std::string p = jstringToStdString(env, path);
    std::string report = imagegen::inspectQnnBinaryReport(p);
    return env->NewStringUTF(report.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeInspectMnnModel(JNIEnv* env,
                                                                       jobject /*thiz*/,
                                                                       jstring path) {
    std::string p = jstringToStdString(env, path);
    std::string report = imagegen::mnnInspectReport(p);
    return env->NewStringUTF(report.c_str());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeRunMnnTextEncode(JNIEnv* env,
                                                                        jobject /*thiz*/,
                                                                        jstring bundleDir,
                                                                        jstring prompt) {
    const std::string dir = jstringToStdString(env, bundleDir);
    const std::string p   = jstringToStdString(env, prompt);
    try {
        // Tokenize.
        imagegen::Tokenizer tok(dir + "/tokenizer.json");
        auto tokens = tok.encode(p);
        __android_log_print(ANDROID_LOG_INFO, kTag,
                            "text encode: prompt=%zu chars → %zu tokens (first=[%d,%d,%d])",
                            p.size(), tokens.size(),
                            tokens.size() > 0 ? tokens[0] : -1,
                            tokens.size() > 1 ? tokens[1] : -1,
                            tokens.size() > 2 ? tokens[2] : -1);

        // Build the [1,77,768] fp32 input embedding (token_emb + pos_emb).
        auto embedding = imagegen::clipEmbedTokens(
            tokens, dir + "/token_emb.bin", dir + "/pos_emb.bin", /*embedDim=*/768);

        // Run MNN forward.
        imagegen::MnnSession sess;
        std::string err;
        if (!sess.load(dir + "/clip_v2.mnn", /*preferOpenCL=*/true, err)) {
            __android_log_print(ANDROID_LOG_ERROR, kTag, "mnn load: %s", err.c_str());
            return nullptr;
        }
        std::vector<float> out;
        if (!sess.runForward(embedding, out, err)) {
            __android_log_print(ANDROID_LOG_ERROR, kTag, "mnn run: %s", err.c_str());
            return nullptr;
        }
        jfloatArray arr = env->NewFloatArray(static_cast<jsize>(out.size()));
        if (!arr) return nullptr;
        env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(out.size()), out.data());
        return arr;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "text-encode exception: %s", e.what());
        return nullptr;
    }
}

// Phase 6 entry point: drive the full diffusion loop with CFG. Returns the
// final latent tensor (fp32, flat 16384 elements). On failure returns a
// jfloatArray of size 0; the multi-line diagnostic is logged at INFO level
// under the "diffusion" tag.
// Phase 7 entry point: full SD 1.5 pipeline → PNG bytes. Returns null on
// failure (logcat tags `imagegen` / `diffusion` carry the diagnostic) or a
// byte[] holding a valid PNG on success. The boot probe writes the output to
// filesDir/sd-debug.png; ImageGenRunner pipes it through a PFD to the widget.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeRunDiffusionPng(JNIEnv* env,
                                                                        jobject /*thiz*/,
                                                                        jstring bundleDir,
                                                                        jstring prompt,
                                                                        jint    iters,
                                                                        jlong   seed) {
    const std::string dir = jstringToStdString(env, bundleDir);
    const std::string p   = jstringToStdString(env, prompt);
    try {
        std::vector<uint8_t> png;
        std::string report;
        const bool ok = imagegen::runDiffusionToPng(dir, p, iters,
                                                    static_cast<uint64_t>(seed),
                                                    png, report);
        std::istringstream is(report);
        for (std::string line; std::getline(is, line); ) {
            __android_log_print(ANDROID_LOG_INFO, kTag, "%s", line.c_str());
        }
        if (!ok || png.empty()) {
            __android_log_print(ANDROID_LOG_WARN, kTag,
                                "nativeRunDiffusionPng: failed (png=%zu)", png.size());
            return nullptr;
        }
        jbyteArray arr = env->NewByteArray(static_cast<jsize>(png.size()));
        if (!arr) return nullptr;
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(png.size()),
                                reinterpret_cast<const jbyte*>(png.data()));
        return arr;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kTag,
                            "runDiffusionToPng exception: %s", e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_nothing_localai_imagegen_NativeImageGen_nativeRunDiffusion(JNIEnv* env,
                                                                     jobject /*thiz*/,
                                                                     jstring bundleDir,
                                                                     jstring prompt,
                                                                     jint    iters,
                                                                     jlong   seed) {
    const std::string dir = jstringToStdString(env, bundleDir);
    const std::string p   = jstringToStdString(env, prompt);
    try {
        std::string report;
        auto latents = imagegen::runDiffusion(dir, p, iters,
                                              static_cast<uint64_t>(seed), report);
        // Always log the report — succeeds or fails, the caller wants the trace.
        std::istringstream is(report);
        for (std::string line; std::getline(is, line); ) {
            __android_log_print(ANDROID_LOG_INFO, kTag, "%s", line.c_str());
        }
        if (latents.empty()) {
            return env->NewFloatArray(0);
        }
        jfloatArray arr = env->NewFloatArray(static_cast<jsize>(latents.size()));
        if (!arr) return nullptr;
        env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(latents.size()),
                                 latents.data());
        return arr;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kTag,
                            "runDiffusion exception: %s", e.what());
        return env->NewFloatArray(0);
    }
}
