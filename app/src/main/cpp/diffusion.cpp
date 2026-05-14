#include "diffusion.hpp"

#include <android/log.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

#include "bundle_loader.hpp"
#include "mnn_session.hpp"
#include "qnn_session.hpp"
#include "scheduler.hpp"
#include "tokenizer.hpp"

namespace imagegen {

namespace {

constexpr const char* kTag           = "diffusion";
constexpr float       kGuidanceScale = 7.5f;
constexpr int         kLatentChan    = 4;
constexpr int         kLatentH       = 64;
constexpr int         kLatentW       = 64;
constexpr int         kLatentElems   = kLatentChan * kLatentH * kLatentW;  // 16384
constexpr int         kEmbedDim      = 768;
constexpr int         kSeqLen        = 77;
constexpr int         kEmbedElems    = kSeqLen * kEmbedDim;  // 59136

// QNN_DATATYPE_* numeric values we care about (mirror QnnTypes.h).
constexpr uint32_t kDtFloat16 = 0x0216;
constexpr uint32_t kDtFloat32 = 0x0232;
constexpr uint32_t kDtInt32   = 0x0032;
constexpr uint32_t kDtUfp16   = 0x0416;   // UFIXED_POINT_16 — xororz w8a16 boundary dtype

uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign  = (x >> 16) & 0x8000u;
    int32_t        exp   = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t       mant  = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half  = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half += 1;  // round-to-nearest-even (approx)
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | (0x1Fu << 10));  // ±inf
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float halfToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            ++exp;
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

// Box-Muller gaussian noise with a deterministic seed.
void fillGaussian(std::vector<float>& out, uint64_t seed) {
    std::mt19937_64 rng(seed);
    // Avoid 0 in u1 to keep log() finite.
    std::uniform_real_distribution<float> uni(1e-7f, 1.0f);
    constexpr float kTwoPi = 6.28318530718f;
    for (std::size_t i = 0; i < out.size(); i += 2) {
        const float u1 = uni(rng);
        const float u2 = uni(rng);
        const float r  = std::sqrt(-2.0f * std::log(u1));
        out[i] = r * std::cos(kTwoPi * u2);
        if (i + 1 < out.size()) out[i + 1] = r * std::sin(kTwoPi * u2);
    }
}

bool containsCi(const std::string& haystack, const char* needle) {
    std::string lh(haystack);
    std::transform(lh.begin(), lh.end(), lh.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::string ln(needle);
    std::transform(ln.begin(), ln.end(), ln.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lh.find(ln) != std::string::npos;
}

// Heuristic: classify a UNet input tensor by name.
enum class UnetInputRole { Latent, Timestep, Embedding, Unknown };
UnetInputRole classifyInput(const std::string& name) {
    if (containsCi(name, "latent") || containsCi(name, "sample") ||
        containsCi(name, "noisy")) return UnetInputRole::Latent;
    if (containsCi(name, "time") || containsCi(name, "step") ||
        name == "t" || name == "T") return UnetInputRole::Timestep;
    if (containsCi(name, "encoder_hidden") || containsCi(name, "context") ||
        containsCi(name, "emb")            || containsCi(name, "prompt") ||
        containsCi(name, "text")) return UnetInputRole::Embedding;
    return UnetInputRole::Unknown;
}

// Encode an fp32 buffer into the byte buffer expected by `info`.
// Honors info.dataType, and for ufp16 (xororz UNet I/O) honors the per-tensor
// scale/offset quantization params extracted from binaryInfo.
bool packFloats(const std::vector<float>& src, const QnnTensorInfo& info,
                std::vector<uint8_t>& dst, std::string& error) {
    const uint32_t dt = info.dataType;
    if (dt == kDtFloat32) {
        dst.resize(src.size() * sizeof(float));
        std::memcpy(dst.data(), src.data(), dst.size());
        return true;
    }
    if (dt == kDtFloat16) {
        dst.resize(src.size() * sizeof(uint16_t));
        auto* p = reinterpret_cast<uint16_t*>(dst.data());
        for (std::size_t i = 0; i < src.size(); ++i) p[i] = floatToHalf(src[i]);
        return true;
    }
    if (dt == kDtUfp16) {
        if (info.quantScale <= 0.0f) {
            error = "packFloats: ufp16 tensor missing scale (got " +
                    std::to_string(info.quantScale) + ")";
            return false;
        }
        // float_value = (q + offset) * scale  →  q = round(f/scale) - offset
        const float invScale = 1.0f / info.quantScale;
        const float off      = static_cast<float>(info.quantOffset);
        dst.resize(src.size() * sizeof(uint16_t));
        auto* p = reinterpret_cast<uint16_t*>(dst.data());
        for (std::size_t i = 0; i < src.size(); ++i) {
            float q = std::nearbyint(src[i] * invScale) - off;
            if (q < 0.0f)     q = 0.0f;
            if (q > 65535.0f) q = 65535.0f;
            p[i] = static_cast<uint16_t>(q);
        }
        return true;
    }
    error = "packFloats: unsupported dtype 0x" + std::to_string(dt);
    return false;
}

bool unpackFloats(const std::vector<uint8_t>& src, const QnnTensorInfo& info,
                  std::size_t numElements, std::vector<float>& dst,
                  std::string& error) {
    const uint32_t dt = info.dataType;
    dst.resize(numElements);
    if (dt == kDtFloat32) {
        if (src.size() < numElements * sizeof(float)) {
            error = "unpackFloats: undersized fp32 buffer";
            return false;
        }
        std::memcpy(dst.data(), src.data(), numElements * sizeof(float));
        return true;
    }
    if (dt == kDtFloat16) {
        if (src.size() < numElements * sizeof(uint16_t)) {
            error = "unpackFloats: undersized fp16 buffer";
            return false;
        }
        const auto* p = reinterpret_cast<const uint16_t*>(src.data());
        for (std::size_t i = 0; i < numElements; ++i) dst[i] = halfToFloat(p[i]);
        return true;
    }
    if (dt == kDtUfp16) {
        if (src.size() < numElements * sizeof(uint16_t)) {
            error = "unpackFloats: undersized ufp16 buffer";
            return false;
        }
        if (info.quantScale <= 0.0f) {
            error = "unpackFloats: ufp16 tensor missing scale";
            return false;
        }
        const auto* p = reinterpret_cast<const uint16_t*>(src.data());
        const float scale = info.quantScale;
        const float off   = static_cast<float>(info.quantOffset);
        for (std::size_t i = 0; i < numElements; ++i) {
            dst[i] = (static_cast<float>(p[i]) + off) * scale;
        }
        return true;
    }
    error = "unpackFloats: unsupported dtype 0x" + std::to_string(dt);
    return false;
}

void logI(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, ap);
    va_end(ap);
}

}  // namespace

std::vector<float> runDiffusion(const std::string& bundleDir,
                                const std::string& prompt,
                                int                iters,
                                uint64_t           seed,
                                std::string&       report) {
    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    std::ostringstream rs;

    // 1. Load bundle manifest.
    Bundle bundle;
    std::string err;
    if (!loadBundle(bundleDir, bundle, err)) { report = "bundle: " + err; return {}; }

    // 2. Tokenize prompt + uncond ("").
    Tokenizer tok(bundle.tokenizerJson);
    auto tokensCond   = tok.encode(prompt);
    auto tokensUncond = tok.encode("");

    // 3. Build CLIP input embeddings (token_emb + pos_emb) and run MNN twice.
    auto embInCond   = clipEmbedTokens(tokensCond,   bundle.tokenEmbBin,
                                       bundle.posEmbBin, kEmbedDim);
    auto embInUncond = clipEmbedTokens(tokensUncond, bundle.tokenEmbBin,
                                       bundle.posEmbBin, kEmbedDim);

    MnnSession mnn;
    if (!mnn.load(bundle.clipMnn, /*preferOpenCL=*/true, err)) {
        report = "mnn load: " + err;
        return {};
    }
    auto t_mnn_loaded = clk::now();
    std::vector<float> embCond, embUncond;
    if (!mnn.runForward(embInCond,   embCond,   err)) { report = "mnn cond: "   + err; return {}; }
    if (!mnn.runForward(embInUncond, embUncond, err)) { report = "mnn uncond: " + err; return {}; }
    if (embCond.size() != kEmbedElems || embUncond.size() != kEmbedElems) {
        report = "mnn output unexpected size: got " + std::to_string(embCond.size()) +
                 " / " + std::to_string(embUncond.size()) +
                 " expected " + std::to_string(kEmbedElems);
        return {};
    }
    auto t_mnn_done = clk::now();
    logI("text encode done (%lld ms)",
         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
             t_mnn_done - t_mnn_loaded).count());

    // 4. QNN session for unet.bin: init + inspect + instantiate.
    QnnSession qnn;
    if (!qnn.initialize(err))                       { report = "qnn init: "        + err; return {}; }
    if (!qnn.inspectBinary(bundle.unetBin, err))    { report = "qnn inspect: "     + err; return {}; }
    if (!qnn.instantiate(err))                      { report = "qnn instantiate: " + err; return {}; }
    auto t_qnn_ready = clk::now();
    logI("qnn instantiate done (%lld ms)",
         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
             t_qnn_ready - t_mnn_done).count());

    if (qnn.graphs().empty()) { report = "unet binary has no graphs"; return {}; }
    const auto& g = qnn.graphs()[0];
    rs << "graph '" << g.name << "' inputs=" << g.inputs.size()
       << " outputs=" << g.outputs.size() << "\n";

    // 5. Identify input slots by name. Crash early if we can't classify.
    int slotLatent = -1, slotTimestep = -1, slotEmb = -1;
    auto describe = [&](const char* role, std::size_t i, const QnnTensorInfo& t) {
        rs << "  " << role << "[" << i << "] '" << t.name
           << "' dt=0x" << std::hex << t.dataType << std::dec << " dims=[";
        for (std::size_t j = 0; j < t.dims.size(); ++j) {
            if (j) rs << 'x';
            rs << t.dims[j];
        }
        rs << "] scale=" << t.quantScale << " offset=" << t.quantOffset << "\n";
    };
    for (std::size_t i = 0; i < g.inputs.size(); ++i) {
        const auto& t = g.inputs[i];
        describe("in", i, t);
        switch (classifyInput(t.name)) {
            case UnetInputRole::Latent:    slotLatent   = (int)i; break;
            case UnetInputRole::Timestep:  slotTimestep = (int)i; break;
            case UnetInputRole::Embedding: slotEmb      = (int)i; break;
            default: break;
        }
    }
    for (std::size_t i = 0; i < g.outputs.size(); ++i) describe("out", i, g.outputs[i]);
    if (slotLatent < 0 || slotTimestep < 0 || slotEmb < 0) {
        report = "could not classify unet inputs (latent=" + std::to_string(slotLatent) +
                 " timestep=" + std::to_string(slotTimestep) +
                 " embedding=" + std::to_string(slotEmb) + ")\n" + rs.str();
        return {};
    }
    if (g.outputs.size() != 1) {
        report = "expected exactly 1 unet output, got " + std::to_string(g.outputs.size());
        return {};
    }
    const auto& outInfo = g.outputs[0];
    const std::size_t outBytes = tensorByteSize(outInfo);
    if (outBytes == 0) {
        report = "unet output dtype unsupported: 0x" +
                 std::to_string(outInfo.dataType);
        return {};
    }

    // 6. Allocate latents as gaussian noise.
    std::vector<float> latents(kLatentElems);
    fillGaussian(latents, seed);

    // 7. Scheduler.
    Scheduler sched;
    sched.setTimesteps(iters);

    // Pre-allocate the byte-buffer holders we'll reuse across iterations.
    std::vector<std::vector<uint8_t>> inBufs(g.inputs.size());
    std::vector<std::vector<uint8_t>> outBufs(1);

    // Pre-pack the embeddings once — they don't change between steps.
    std::vector<uint8_t> embCondBytes, embUncondBytes;
    if (!packFloats(embCond,   g.inputs[slotEmb], embCondBytes,   err)) {
        report = "pack embCond: " + err; return {};
    }
    if (!packFloats(embUncond, g.inputs[slotEmb], embUncondBytes, err)) {
        report = "pack embUncond: " + err; return {};
    }

    // Helper: run one UNet pass for the given timestep + embedding bytes.
    auto runUnet = [&](const std::vector<float>& latentFp32,
                       int t,
                       const std::vector<uint8_t>& embBytes,
                       std::vector<float>& noiseOut,
                       std::string& runErr) -> bool {
        if (!packFloats(latentFp32, g.inputs[slotLatent],
                        inBufs[slotLatent], runErr)) return false;

        // Timestep: pack a single value. Honor the slot's dtype/rank.
        const auto& tsInfo = g.inputs[slotTimestep];
        std::size_t tsElems = 1;
        for (uint32_t d : tsInfo.dims) tsElems *= d;
        std::vector<float> tsFp32(tsElems, static_cast<float>(t));
        if (tsInfo.dataType == kDtInt32) {
            inBufs[slotTimestep].resize(tsElems * sizeof(int32_t));
            auto* p = reinterpret_cast<int32_t*>(inBufs[slotTimestep].data());
            for (std::size_t i = 0; i < tsElems; ++i) p[i] = t;
        } else if (!packFloats(tsFp32, tsInfo,
                               inBufs[slotTimestep], runErr)) {
            return false;
        }
        inBufs[slotEmb] = embBytes;

        outBufs[0].assign(outBytes, 0);
        if (!qnn.execute(0, inBufs, outBufs, runErr)) return false;
        return unpackFloats(outBufs[0], outInfo, kLatentElems,
                            noiseOut, runErr);
    };

    // 8. Diffusion loop with classifier-free guidance.
    std::vector<float> noiseCond, noiseUncond;
    std::vector<float> pred(kLatentElems);
    auto t_loop_start = clk::now();
    for (int i = 0; i < iters; ++i) {
        const int t = sched.timestep(i);
        if (!runUnet(latents, t, embCondBytes,   noiseCond,   err)) {
            report = "unet step " + std::to_string(i) + " cond: " + err + "\n" + rs.str();
            return {};
        }
        if (!runUnet(latents, t, embUncondBytes, noiseUncond, err)) {
            report = "unet step " + std::to_string(i) + " uncond: " + err + "\n" + rs.str();
            return {};
        }
        for (int k = 0; k < kLatentElems; ++k) {
            pred[k] = noiseUncond[k] + kGuidanceScale * (noiseCond[k] - noiseUncond[k]);
        }
        latents = sched.step(pred, i, latents);

        // Sanity check: every 4 steps, log a min/max so we can spot divergence early.
        float lo = latents[0], hi = latents[0];
        for (float v : latents) { lo = std::min(lo, v); hi = std::max(hi, v); }
        logI("step %d/%d (t=%d) latents[min=%.3f max=%.3f]", i + 1, iters, t, lo, hi);
        if (!std::isfinite(lo) || !std::isfinite(hi)) {
            report = "latents went non-finite at step " + std::to_string(i) +
                     "\n" + rs.str();
            return {};
        }
    }
    auto t_loop_end = clk::now();

    // 9. Compose success report.
    rs << "iters=" << iters << " seed=" << seed << "\n"
       << "tokens.cond=" << tokensCond.size() << " tokens.uncond=" << tokensUncond.size() << "\n"
       << "mnn backend=" << mnn.backendName() << "\n"
       << "qnn iface=" << qnn.interfaceVersion() << "\n"
       << "timing: load+mnn=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_mnn_done - t_start).count() << "ms"
       <<       " qnn_init=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_qnn_ready - t_mnn_done).count() << "ms"
       <<       " loop="     << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_loop_end - t_loop_start).count() << "ms"
       <<       " total="    << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_loop_end - t_start).count() << "ms\n";

    rs << "latent first8 = [";
    for (int k = 0; k < 8 && k < kLatentElems; ++k) {
        if (k) rs << ", ";
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.4f", latents[k]);
        rs << buf;
    }
    rs << "]\n";

    report = rs.str();
    return latents;
}

// -----------------------------------------------------------------------------
// Phase 7: VAE decode + PNG encode.
// -----------------------------------------------------------------------------

// Declared in png_encode.cpp.
bool encodeRgbToPng(int width, int height,
                    const std::vector<uint8_t>& rgb,
                    std::vector<uint8_t>& pngOut,
                    std::string& error);

namespace {

// SD 1.5 VAE latent scaling factor — the encoder multiplies by 0.18215, the
// decoder undoes it.
constexpr float kLatentScale = 0.18215f;

// Transpose flat [C,H,W] → flat [H,W,C]. All buffers row-major.
void chwToHwc(const std::vector<float>& chw, std::vector<float>& hwc,
              int C, int H, int W) {
    hwc.resize(static_cast<std::size_t>(C) * H * W);
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                hwc[(static_cast<std::size_t>(h) * W + w) * C + c] =
                    chw[(static_cast<std::size_t>(c) * H + h) * W + w];
            }
        }
    }
}

// Classify a 4D VAE tensor as NCHW or NHWC based on which axis matches the
// expected channel count. Returns true if classified; fills H,W and `isNhwc`.
bool classifyLayout4D(const std::vector<uint32_t>& dims,
                      uint32_t expectedChannels,
                      int& H, int& W, bool& isNhwc) {
    if (dims.size() != 4) return false;
    if (dims[1] == expectedChannels) {
        isNhwc = false;
        H = static_cast<int>(dims[2]);
        W = static_cast<int>(dims[3]);
        return true;
    }
    if (dims[3] == expectedChannels) {
        isNhwc = true;
        H = static_cast<int>(dims[1]);
        W = static_cast<int>(dims[2]);
        return true;
    }
    return false;
}

}  // namespace

bool vaeDecodeToPng(const std::string&        bundleDir,
                    const std::vector<float>& latents,
                    std::vector<uint8_t>&     pngOut,
                    std::string&              report) {
    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    std::ostringstream rs;

    Bundle bundle;
    std::string err;
    if (!loadBundle(bundleDir, bundle, err)) {
        report = "vae bundle: " + err;
        return false;
    }
    if (latents.size() != kLatentElems) {
        report = "vaeDecodeToPng: latents size " + std::to_string(latents.size()) +
                 " != expected " + std::to_string(kLatentElems);
        return false;
    }

    // Fresh QnnSession for the VAE — the UNet session has already been
    // destroyed by the time runDiffusion returned, so this is the only QNN
    // context alive on the chip.
    QnnSession qnn;
    if (!qnn.initialize(err))                          { report = "vae init: " + err; return false; }
    if (!qnn.inspectBinary(bundle.vaeDecoderBin, err)) { report = "vae inspect: " + err; return false; }
    if (!qnn.instantiate(err))                         { report = "vae instantiate: " + err; return false; }
    auto t_loaded = clk::now();

    if (qnn.graphs().empty()) { report = "vae binary has no graphs"; return false; }
    const auto& g = qnn.graphs()[0];
    if (g.inputs.size() != 1 || g.outputs.size() != 1) {
        report = "vae expects 1 in/1 out, got in=" + std::to_string(g.inputs.size()) +
                 " out=" + std::to_string(g.outputs.size());
        return false;
    }
    const auto& inInfo  = g.inputs[0];
    const auto& outInfo = g.outputs[0];

    // Classify input layout (expect 4 channels for SD latent).
    int  inH = 0, inW = 0;
    bool inIsNhwc = false;
    if (!classifyLayout4D(inInfo.dims, 4u, inH, inW, inIsNhwc)) {
        report = "vae input dims could not classify NCHW/NHWC for 4-channel latent: '" +
                 inInfo.name + "' rank=" + std::to_string(inInfo.dims.size());
        return false;
    }
    if (inH != kLatentH || inW != kLatentW) {
        report = "vae input shape mismatch: got H=" + std::to_string(inH) +
                 " W=" + std::to_string(inW) +
                 " expected " + std::to_string(kLatentH) + "x" + std::to_string(kLatentW);
        return false;
    }

    // Apply SD 1.5 latent scale. `runDiffusion` produces latents in whatever
    // layout the UNet uses; xororz exports UNet+VAE consistently so we assume
    // the layouts match (the dims are identical: 1x4x64x64 in NCHW or 1x64x64x4
    // in NHWC — both have the same byte order if both models agree).
    std::vector<float> scaled(latents.size());
    const float invScale = 1.0f / kLatentScale;
    for (std::size_t i = 0; i < latents.size(); ++i) {
        scaled[i] = latents[i] * invScale;
    }

    std::vector<uint8_t> inBytes;
    if (!packFloats(scaled, inInfo, inBytes, err)) {
        report = "vae pack input: " + err;
        return false;
    }

    // Classify output layout (expect 3 channels for RGB).
    int  outH = 0, outW = 0;
    bool outIsNhwc = false;
    if (!classifyLayout4D(outInfo.dims, 3u, outH, outW, outIsNhwc)) {
        report = "vae output dims could not classify NCHW/NHWC for 3-channel RGB: '" +
                 outInfo.name + "' rank=" + std::to_string(outInfo.dims.size());
        return false;
    }
    const std::size_t outElems = static_cast<std::size_t>(3) * outH * outW;

    std::vector<std::vector<uint8_t>> inBufs(1);
    inBufs[0] = std::move(inBytes);
    std::vector<std::vector<uint8_t>> outBufs(1);
    outBufs[0].assign(tensorByteSize(outInfo), 0);

    auto t_exec_start = clk::now();
    if (!qnn.execute(0, inBufs, outBufs, err)) {
        report = "vae execute: " + err;
        return false;
    }
    auto t_exec_end = clk::now();

    std::vector<float> outFp32;
    if (!unpackFloats(outBufs[0], outInfo, outElems, outFp32, err)) {
        report = "vae unpack: " + err;
        return false;
    }

    // Reorder to HWC interleaved if necessary.
    std::vector<float> hwc;
    const std::vector<float>* hwcView = nullptr;
    if (outIsNhwc) {
        hwcView = &outFp32;
    } else {
        chwToHwc(outFp32, hwc, /*C=*/3, outH, outW);
        hwcView = &hwc;
    }

    // Denorm: SD VAE outputs in roughly [-1, 1]. Clamp, shift, scale, cast.
    std::vector<uint8_t> rgb(outElems);
    int nonFinite = 0;
    float mn = (*hwcView)[0], mx = (*hwcView)[0];
    for (std::size_t i = 0; i < outElems; ++i) {
        float v = (*hwcView)[i];
        if (!std::isfinite(v)) { v = 0.0f; ++nonFinite; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        v = std::min(1.0f, std::max(-1.0f, v));
        const float u = (v + 1.0f) * 127.5f;
        rgb[i] = static_cast<uint8_t>(u < 0.0f ? 0 : (u > 255.0f ? 255 : u));
    }

    if (!encodeRgbToPng(outW, outH, rgb, pngOut, err)) {
        report = "png encode: " + err;
        return false;
    }

    auto t_end = clk::now();
    rs << "vae " << (outIsNhwc ? "NHWC" : "NCHW") << " "
       << outH << "x" << outW
       << " range[" << mn << "," << mx << "]"
       << (nonFinite > 0 ? (" nonFinite=" + std::to_string(nonFinite)) : "")
       << " pngBytes=" << pngOut.size()
       << " timing: load=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_loaded - t_start).count() << "ms"
       <<       " exec=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_exec_end - t_exec_start).count() << "ms"
       <<       " total=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_end - t_start).count() << "ms\n";

    report = rs.str();
    return true;
}

bool runDiffusionToPng(const std::string&    bundleDir,
                       const std::string&    prompt,
                       int                   iters,
                       uint64_t              seed,
                       std::vector<uint8_t>& pngOut,
                       std::string&          report) {
    std::string diffReport;
    std::vector<float> latents = runDiffusion(bundleDir, prompt, iters, seed, diffReport);
    report = diffReport;
    if (latents.empty()) {
        return false;
    }
    std::string vaeReport;
    if (!vaeDecodeToPng(bundleDir, latents, pngOut, vaeReport)) {
        report += vaeReport;
        return false;
    }
    report += vaeReport;
    return true;
}

}  // namespace imagegen
