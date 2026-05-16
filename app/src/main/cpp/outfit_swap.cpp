// Outfit-swap pipeline orchestrator (PLAN-OUTFIT-SWAP.md Phase 2/3).
//
// Self-contained: reuses Bundle/Tokenizer/Mnn/Qnn/Scheduler helpers via their
// public headers, but duplicates the small pack/unpack and shape-classify
// helpers from diffusion.cpp rather than refactoring those into a shared
// header. The inpaint diffusion loop is structurally similar to runDiffusion
// but the UNet input is 9-channel and the mask/masked-image latents are
// precomputed once.

#include "outfit_swap.hpp"

#include <android/log.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <random>
#include <sstream>

#include "bundle_loader.hpp"
#include "diffusion.hpp"          // vaeDecodeToPng — reused as Stage 7
#include "mask_ops.hpp"
#include "mnn_session.hpp"
#include "png_encode.hpp"
#include "qnn_session.hpp"
#include "scheduler.hpp"
#include "segformer.hpp"
#include "tokenizer.hpp"

namespace imagegen {

namespace {

constexpr const char* kTag = "outfit_swap";

constexpr int   kImgC        = 3;
constexpr int   kImgH        = 512;
constexpr int   kImgW        = 512;
constexpr int   kLatentC     = 4;
constexpr int   kLatentH     = 64;
constexpr int   kLatentW     = 64;
constexpr int   kLatentElems = kLatentC * kLatentH * kLatentW;       // 16384
constexpr int   kUnetInC     = 9;
constexpr int   kUnetInElems = kUnetInC * kLatentH * kLatentW;        // 36864
constexpr int   kEmbedDim    = 768;
constexpr int   kSeqLen      = 77;
constexpr int   kEmbedElems  = kSeqLen * kEmbedDim;                   // 59136

constexpr float kVaeLatentScale = 0.18215f;     // SD 1.5 convention
constexpr float kGuidanceScale  = 7.5f;

// QNN_DATATYPE_* numeric values (mirror QnnTypes.h).
constexpr uint32_t kDtFloat16 = 0x0216;
constexpr uint32_t kDtFloat32 = 0x0232;
constexpr uint32_t kDtInt32   = 0x0032;
constexpr uint32_t kDtUfp16   = 0x0416;

// ------------------------------ small helpers ------------------------------

uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t        exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t       mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half  = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half += 1;
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | (0x1Fu << 10));
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float halfToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
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
    float r; std::memcpy(&r, &f, sizeof(r));
    return r;
}

bool packFloats(const std::vector<float>& src, const QnnTensorInfo& info,
                std::vector<uint8_t>& bytes, std::string& error) {
    const std::size_t elems = src.size();
    const std::size_t outBytes = tensorByteSize(info);
    bytes.assign(outBytes, 0);
    switch (info.dataType) {
        case kDtFloat32:
            if (outBytes != elems * sizeof(float)) {
                error = "packFloats: fp32 size mismatch"; return false;
            }
            std::memcpy(bytes.data(), src.data(), outBytes);
            return true;
        case kDtFloat16: {
            if (outBytes != elems * 2) { error = "packFloats: fp16 size mismatch"; return false; }
            auto* p = reinterpret_cast<uint16_t*>(bytes.data());
            for (std::size_t i = 0; i < elems; ++i) p[i] = floatToHalf(src[i]);
            return true;
        }
        case kDtUfp16: {
            if (info.quantScale == 0.0f) { error = "packFloats: ufp16 missing scale"; return false; }
            if (outBytes != elems * 2)   { error = "packFloats: ufp16 size mismatch"; return false; }
            const float inv = 1.0f / info.quantScale;
            auto* p = reinterpret_cast<uint16_t*>(bytes.data());
            for (std::size_t i = 0; i < elems; ++i) {
                int32_t q = static_cast<int32_t>(std::nearbyint(src[i] * inv)) - info.quantOffset;
                if (q < 0) q = 0;
                if (q > 0xFFFF) q = 0xFFFF;
                p[i] = static_cast<uint16_t>(q);
            }
            return true;
        }
        default:
            error = "packFloats: unsupported dtype 0x" + std::to_string(info.dataType);
            return false;
    }
}

bool unpackFloats(const std::vector<uint8_t>& src, const QnnTensorInfo& info,
                  std::size_t expected, std::vector<float>& out, std::string& error) {
    out.assign(expected, 0.0f);
    switch (info.dataType) {
        case kDtFloat32:
            if (src.size() < expected * sizeof(float)) {
                error = "unpackFloats: undersized fp32 buffer"; return false;
            }
            std::memcpy(out.data(), src.data(), expected * sizeof(float));
            return true;
        case kDtFloat16: {
            if (src.size() < expected * 2) { error = "unpackFloats: undersized fp16 buffer"; return false; }
            const auto* p = reinterpret_cast<const uint16_t*>(src.data());
            for (std::size_t i = 0; i < expected; ++i) out[i] = halfToFloat(p[i]);
            return true;
        }
        case kDtUfp16: {
            if (src.size() < expected * 2)  { error = "unpackFloats: undersized ufp16 buffer"; return false; }
            if (info.quantScale == 0.0f)    { error = "unpackFloats: ufp16 missing scale"; return false; }
            const auto* p = reinterpret_cast<const uint16_t*>(src.data());
            const float s = info.quantScale;
            const int32_t off = info.quantOffset;
            for (std::size_t i = 0; i < expected; ++i) {
                out[i] = (static_cast<float>(p[i]) + static_cast<float>(off)) * s;
            }
            return true;
        }
        default:
            error = "unpackFloats: unsupported dtype 0x" + std::to_string(info.dataType);
            return false;
    }
}

bool classify4D(const std::vector<uint32_t>& dims, uint32_t expectedChannels,
                int& H, int& W, bool& isNhwc) {
    if (dims.size() != 4) return false;
    if (dims[1] == expectedChannels) {
        isNhwc = false; H = (int)dims[2]; W = (int)dims[3]; return true;
    }
    if (dims[3] == expectedChannels) {
        isNhwc = true;  H = (int)dims[1]; W = (int)dims[2]; return true;
    }
    return false;
}

// Inpaint UNet input-slot classifier. Names from the diffusers convention
// (the compile script passes input_specs={"sample", "timestep", "encoder_hidden_states"}).
enum class UnetSlot { Sample, Timestep, EncoderHiddenStates, Unknown };

UnetSlot classifyUnetSlot(const std::string& name) {
    const std::string n = [&]{
        std::string s = name;
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }();
    if (n.find("sample") != std::string::npos || n.find("latent") != std::string::npos) {
        return UnetSlot::Sample;
    }
    if (n.find("timestep") != std::string::npos || n.find("time") != std::string::npos) {
        return UnetSlot::Timestep;
    }
    if (n.find("encoder_hidden_states") != std::string::npos ||
        n.find("hidden")                != std::string::npos ||
        n.find("text")                  != std::string::npos ||
        n.find("embed")                 != std::string::npos) {
        return UnetSlot::EncoderHiddenStates;
    }
    return UnetSlot::Unknown;
}

// Nearest-neighbor upsample of a uint8 classmap from src(srcH,srcW) → dst(dstH,dstW).
void upsampleClassmapNearest(const std::vector<uint8_t>& src, int srcH, int srcW,
                             std::vector<uint8_t>& dst, int dstH, int dstW) {
    dst.assign(static_cast<std::size_t>(dstH) * dstW, 0);
    if (srcH == dstH && srcW == dstW) { dst = src; return; }
    for (int y = 0; y < dstH; ++y) {
        const int sy = std::min(srcH - 1, y * srcH / dstH);
        for (int x = 0; x < dstW; ++x) {
            const int sx = std::min(srcW - 1, x * srcW / dstW);
            dst[static_cast<std::size_t>(y) * dstW + x] =
                src[static_cast<std::size_t>(sy) * srcW + sx];
        }
    }
}

// VAE-input normalization: [0,1] RGB CHW → [-1,1] RGB CHW (SD 1.5 convention).
void vaeNormalizeIn(const std::vector<float>& raw, std::vector<float>& out) {
    out.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = raw[i] * 2.0f - 1.0f;
    }
}

// CHW → HWC transpose.
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

void hwcToChw(const std::vector<float>& hwc, std::vector<float>& chw,
              int C, int H, int W) {
    chw.resize(static_cast<std::size_t>(C) * H * W);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                chw[(static_cast<std::size_t>(c) * H + h) * W + w] =
                    hwc[(static_cast<std::size_t>(h) * W + w) * C + c];
            }
        }
    }
}

void fillGaussian(std::vector<float>& out, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& v : out) v = dist(rng);
}

void logI(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, ap);
    va_end(ap);
}

// ------------------------- VAE encode -------------------------
// Load vae_encoder.bin, run the [1,3,512,512] → [1,4,64,64] forward, scale by
// 0.18215 (SD convention). Returns the CHW fp32 latent (16384 elements).
bool vaeEncodeImage(const std::string& vaeEncoderBin,
                    const std::vector<float>& rawRgb01CHW,
                    std::vector<float>& latentChwOut,
                    std::string& report) {
    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    std::ostringstream rs;

    if (static_cast<int>(rawRgb01CHW.size()) != kImgC * kImgH * kImgW) {
        rs << "vaeEnc: input size mismatch " << rawRgb01CHW.size();
        report = rs.str(); return false;
    }
    std::vector<float> normalized;
    vaeNormalizeIn(rawRgb01CHW, normalized);

    QnnSession qnn;
    std::string err;
    if (!qnn.initialize(err))                       { report = "vaeEnc init: " + err; return false; }
    if (!qnn.inspectBinary(vaeEncoderBin, err))     { report = "vaeEnc inspect: " + err; return false; }
    if (!qnn.instantiate(err))                      { report = "vaeEnc instantiate: " + err; return false; }
    auto t_loaded = clk::now();

    if (qnn.graphs().empty()) { report = "vaeEnc: no graphs"; return false; }
    const auto& g = qnn.graphs()[0];
    if (g.inputs.size() != 1 || g.outputs.size() != 1) {
        report = "vaeEnc expects 1in/1out got in=" + std::to_string(g.inputs.size()); return false;
    }
    const auto& inInfo  = g.inputs[0];
    const auto& outInfo = g.outputs[0];

    int inH = 0, inW = 0; bool inIsNhwc = false;
    if (!classify4D(inInfo.dims, kImgC, inH, inW, inIsNhwc)) {
        report = "vaeEnc input dims unclassifiable"; return false;
    }
    if (inH != kImgH || inW != kImgW) {
        report = "vaeEnc input shape mismatch"; return false;
    }
    const std::vector<float>* packView = &normalized;
    std::vector<float> tmpHwc;
    if (inIsNhwc) {
        chwToHwc(normalized, tmpHwc, kImgC, kImgH, kImgW);
        packView = &tmpHwc;
    }
    std::vector<uint8_t> inBytes;
    if (!packFloats(*packView, inInfo, inBytes, err)) {
        report = "vaeEnc pack: " + err; return false;
    }

    int outH = 0, outW = 0; bool outIsNhwc = false;
    if (!classify4D(outInfo.dims, kLatentC, outH, outW, outIsNhwc)) {
        report = "vaeEnc output dims unclassifiable"; return false;
    }
    if (outH != kLatentH || outW != kLatentW) {
        report = "vaeEnc output shape mismatch"; return false;
    }
    const std::size_t outElems = static_cast<std::size_t>(kLatentC) * outH * outW;

    std::vector<std::vector<uint8_t>> inBufs(1);
    inBufs[0] = std::move(inBytes);
    std::vector<std::vector<uint8_t>> outBufs(1);
    outBufs[0].assign(tensorByteSize(outInfo), 0);

    auto t_exec_start = clk::now();
    if (!qnn.execute(0, inBufs, outBufs, err)) {
        report = "vaeEnc execute: " + err; return false;
    }
    auto t_exec_end = clk::now();

    std::vector<float> outFp32;
    if (!unpackFloats(outBufs[0], outInfo, outElems, outFp32, err)) {
        report = "vaeEnc unpack: " + err; return false;
    }
    if (outIsNhwc) {
        hwcToChw(outFp32, latentChwOut, kLatentC, outH, outW);
    } else {
        latentChwOut = std::move(outFp32);
    }
    // Scale by SD's latent factor (encoder outputs are pre-scale).
    for (float& v : latentChwOut) v *= kVaeLatentScale;

    auto t_end = clk::now();
    rs << "vaeEnc " << (inIsNhwc ? "NHWC" : "NCHW")
       << "→" << (outIsNhwc ? "NHWC" : "NCHW")
       << " load=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_loaded - t_start).count() << "ms"
       << " exec=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_exec_end - t_exec_start).count() << "ms"
       << " total=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms";
    report = rs.str();
    return true;
}

// ------------------------- inpaint UNet diffusion -------------------------
// Build the 9-channel UNet input each step. `zT` is the current 4-ch noisy
// latent; mask (1-ch) and maskedImg (4-ch) are precomputed constants. Output
// is flat fp32 of size 9*64*64. Layout is CHW for now; the slot dtype/
// transpose to NHWC if needed is handled by packFloats + a separate transpose.
void buildUnetInputChw(const std::vector<float>& zT,
                       const std::vector<float>& mask64,
                       const std::vector<float>& maskedImg,
                       std::vector<float>& chwOut) {
    const std::size_t plane = static_cast<std::size_t>(kLatentH) * kLatentW;
    chwOut.assign(static_cast<std::size_t>(kUnetInC) * plane, 0.0f);
    // Channels 0..3 : z_t (noisy latent).
    std::memcpy(chwOut.data(), zT.data(), 4 * plane * sizeof(float));
    // Channel 4 : mask.
    std::memcpy(chwOut.data() + 4 * plane, mask64.data(), plane * sizeof(float));
    // Channels 5..8 : masked image latent.
    std::memcpy(chwOut.data() + 5 * plane, maskedImg.data(), 4 * plane * sizeof(float));
}

// Run the full inpaint UNet DDIM loop with CFG. The mask + masked-image
// latents are constant; we re-pack them per step only if the slot's expected
// byte layout is NHWC (in which case we need to interleave with the changing
// z_t). For NCHW, mask+maskedImg occupy a contiguous suffix of the input
// buffer so we precompute those bytes once.
bool runInpaintDiffusion(const std::string& inpaintUnetBin,
                         const std::vector<float>& embCond,
                         const std::vector<float>& embUncond,
                         const std::vector<float>& imageLatent4ch,
                         const std::vector<float>& mask64,
                         int iterations, uint64_t seed,
                         StepCallback onStep,
                         std::vector<float>& finalLatent,
                         std::string& report) {
    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    std::ostringstream rs;

    if (imageLatent4ch.size() != static_cast<std::size_t>(kLatentElems)) {
        report = "inpaint: imageLatent size mismatch"; return false;
    }
    if (mask64.size() != static_cast<std::size_t>(kLatentH) * kLatentW) {
        report = "inpaint: mask64 size mismatch"; return false;
    }

    // Precompute masked image latent = image * (1 - mask).
    std::vector<float> maskedImg;
    buildMaskedImageLatent(imageLatent4ch, mask64, kLatentH, kLatentW, maskedImg);

    QnnSession qnn;
    std::string err;
    if (!qnn.initialize(err))                      { report = "inp init: " + err; return false; }
    if (!qnn.inspectBinary(inpaintUnetBin, err))   { report = "inp inspect: " + err; return false; }
    if (!qnn.instantiate(err))                     { report = "inp instantiate: " + err; return false; }
    auto t_qnn_ready = clk::now();
    if (qnn.graphs().empty()) { report = "inp: no graphs"; return false; }
    const auto& g = qnn.graphs()[0];

    int slotSample = -1, slotTimestep = -1, slotEmb = -1;
    for (std::size_t i = 0; i < g.inputs.size(); ++i) {
        switch (classifyUnetSlot(g.inputs[i].name)) {
            case UnetSlot::Sample:              slotSample   = (int)i; break;
            case UnetSlot::Timestep:            slotTimestep = (int)i; break;
            case UnetSlot::EncoderHiddenStates: slotEmb      = (int)i; break;
            default: break;
        }
    }
    if (slotSample < 0 || slotTimestep < 0 || slotEmb < 0) {
        report = "inp: could not classify slots "
                 "(sample=" + std::to_string(slotSample) +
                 " ts=" + std::to_string(slotTimestep) +
                 " emb=" + std::to_string(slotEmb) + ")";
        return false;
    }
    if (g.outputs.size() != 1) {
        report = "inp: expected 1 output got " + std::to_string(g.outputs.size());
        return false;
    }

    // Validate sample input shape: must be 9-channel 64×64.
    const auto& sampleInfo = g.inputs[slotSample];
    int sH = 0, sW = 0; bool sIsNhwc = false;
    if (!classify4D(sampleInfo.dims, kUnetInC, sH, sW, sIsNhwc)) {
        report = "inp: sample input not classifiable as 9-channel "
                 "(rank=" + std::to_string(sampleInfo.dims.size()) + ")";
        return false;
    }
    if (sH != kLatentH || sW != kLatentW) {
        report = "inp: sample input shape mismatch H=" + std::to_string(sH);
        return false;
    }

    const auto& outInfo = g.outputs[0];
    int oH = 0, oW = 0; bool oIsNhwc = false;
    if (!classify4D(outInfo.dims, kLatentC, oH, oW, oIsNhwc)) {
        report = "inp: output not classifiable as 4-channel"; return false;
    }
    const std::size_t outBytes = tensorByteSize(outInfo);

    // Pre-pack the text embeddings.
    std::vector<uint8_t> embCondBytes, embUncondBytes;
    if (!packFloats(embCond,   g.inputs[slotEmb], embCondBytes,   err)) {
        report = "inp pack embCond: " + err; return false;
    }
    if (!packFloats(embUncond, g.inputs[slotEmb], embUncondBytes, err)) {
        report = "inp pack embUncond: " + err; return false;
    }

    // Noise init.
    std::vector<float> latents(kLatentElems);
    fillGaussian(latents, seed);

    Scheduler sched;
    sched.setTimesteps(iterations);

    std::vector<std::vector<uint8_t>> inBufs(g.inputs.size());
    std::vector<std::vector<uint8_t>> outBufs(1);

    auto runStep = [&](const std::vector<float>& zT, int t,
                       const std::vector<uint8_t>& embBytes,
                       std::vector<float>& noiseOut, std::string& runErr) -> bool {
        std::vector<float> unetInChw;
        buildUnetInputChw(zT, mask64, maskedImg, unetInChw);
        std::vector<float> unetIn;
        if (sIsNhwc) {
            chwToHwc(unetInChw, unetIn, kUnetInC, kLatentH, kLatentW);
        } else {
            unetIn = std::move(unetInChw);
        }
        if (!packFloats(unetIn, sampleInfo, inBufs[slotSample], runErr)) return false;

        const auto& tsInfo = g.inputs[slotTimestep];
        std::size_t tsElems = 1;
        for (uint32_t d : tsInfo.dims) tsElems *= d;
        if (tsInfo.dataType == kDtInt32) {
            inBufs[slotTimestep].resize(tsElems * sizeof(int32_t));
            auto* p = reinterpret_cast<int32_t*>(inBufs[slotTimestep].data());
            for (std::size_t i = 0; i < tsElems; ++i) p[i] = t;
        } else {
            std::vector<float> tsFp32(tsElems, static_cast<float>(t));
            if (!packFloats(tsFp32, tsInfo, inBufs[slotTimestep], runErr)) return false;
        }
        inBufs[slotEmb] = embBytes;

        outBufs[0].assign(outBytes, 0);
        if (!qnn.execute(0, inBufs, outBufs, runErr)) return false;

        std::vector<float> noiseRaw;
        if (!unpackFloats(outBufs[0], outInfo, kLatentElems, noiseRaw, runErr)) return false;
        if (oIsNhwc) {
            hwcToChw(noiseRaw, noiseOut, kLatentC, oH, oW);
        } else {
            noiseOut = std::move(noiseRaw);
        }
        return true;
    };

    std::vector<float> noiseCond, noiseUncond;
    std::vector<float> pred(kLatentElems);
    auto t_loop_start = clk::now();
    for (int i = 0; i < iterations; ++i) {
        const int t = sched.timestep(i);
        if (!runStep(latents, t, embCondBytes, noiseCond, err)) {
            report = "inp step " + std::to_string(i) + " cond: " + err; return false;
        }
        if (!runStep(latents, t, embUncondBytes, noiseUncond, err)) {
            report = "inp step " + std::to_string(i) + " uncond: " + err; return false;
        }
        for (int k = 0; k < kLatentElems; ++k) {
            pred[k] = noiseUncond[k] + kGuidanceScale * (noiseCond[k] - noiseUncond[k]);
        }
        latents = sched.step(pred, i, latents);
        if (onStep) onStep(i + 1, iterations);

        float lo = latents[0], hi = latents[0];
        for (float v : latents) { lo = std::min(lo, v); hi = std::max(hi, v); }
        logI("inp step %d/%d t=%d latents[min=%.3f max=%.3f]", i + 1, iterations, t, lo, hi);
        if (!std::isfinite(lo) || !std::isfinite(hi)) {
            report = "inp: latents non-finite at step " + std::to_string(i);
            return false;
        }
    }
    auto t_loop_end = clk::now();

    finalLatent = std::move(latents);
    rs << "inp ok iters=" << iterations
       << " qnn_init=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_qnn_ready - t_start).count() << "ms"
       << " loop="     << std::chrono::duration_cast<std::chrono::milliseconds>(t_loop_end - t_loop_start).count() << "ms";
    report = rs.str();
    return true;
}

}  // namespace

bool runOutfitSwap(const OutfitSwapParams& params,
                   StageCallback           onStage,
                   StepCallback            onStep,
                   std::vector<uint8_t>&   pngOut,
                   std::string&            report) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    std::ostringstream rs;

    if (static_cast<int>(params.rawRgbFp32.size()) != kImgC * kImgH * kImgW) {
        report = "rawRgbFp32 size " + std::to_string(params.rawRgbFp32.size()) +
                 " != " + std::to_string(kImgC * kImgH * kImgW);
        return false;
    }

    // -------- Stage 1: SegFormer → classmap. --------
    if (onStage) onStage("segmenting");
    std::vector<uint8_t> classmapLow;
    int segH = 0, segW = 0;
    std::string segReport;
    if (!runSegformerClassmap(params.segformerBin, params.rawRgbFp32,
                              classmapLow, segH, segW, segReport)) {
        report = "stage segmenting: " + segReport; return false;
    }
    rs << "[seg] " << segReport << "\n";

    // Upsample classmap to 512×512 for the mask chain.
    std::vector<uint8_t> classmap;
    upsampleClassmapNearest(classmapLow, segH, segW, classmap, kImgH, kImgW);

    // -------- Stage 2: mask chain. --------
    std::vector<uint8_t> binMask;
    classmapToBinaryMask(classmap, kImgH, kImgW, params.selectedClasses, binMask);
    std::vector<uint8_t> dilated;
    dilateMask5px(binMask, kImgH, kImgW, dilated);
    std::vector<float> feathered;
    featherMaskGaussian(dilated, kImgH, kImgW, /*sigma=*/2.0f, feathered);
    std::vector<float> mask64;
    downsampleMaskMean(feathered, kImgH, kImgW, /*factor=*/8, mask64);

    int maskHits = 0;
    for (float v : mask64) if (v > 0.5f) ++maskHits;
    rs << "[mask] 64x64 hits=" << maskHits << "/" << mask64.size() << "\n";
    if (maskHits == 0) {
        report = rs.str() + "stage mask: classmap had no pixels matching selectedClasses="
                 + std::to_string(params.selectedClasses);
        return false;
    }

    // -------- Stage 3+4: text encode + VAE encode (encoding stage). --------
    if (onStage) onStage("encoding");

    Bundle bundle;
    std::string err;
    if (!loadBundle(params.bundleDir, bundle, err)) {
        report = rs.str() + "bundle: " + err; return false;
    }

    // CLIP text encode (cond + uncond), reusing MNN bundle.
    Tokenizer tok(bundle.tokenizerJson);
    auto tokensCond   = tok.encode(params.prompt);
    auto tokensUncond = tok.encode("");
    auto embInCond   = clipEmbedTokens(tokensCond,   bundle.tokenEmbBin,
                                       bundle.posEmbBin, kEmbedDim);
    auto embInUncond = clipEmbedTokens(tokensUncond, bundle.tokenEmbBin,
                                       bundle.posEmbBin, kEmbedDim);

    MnnSession mnn;
    if (!mnn.load(bundle.clipMnn, /*preferOpenCL=*/true, err)) {
        report = rs.str() + "mnn load: " + err; return false;
    }
    std::vector<float> embCond, embUncond;
    if (!mnn.runForward(embInCond,   embCond,   err)) { report = rs.str() + "mnn cond: "   + err; return false; }
    if (!mnn.runForward(embInUncond, embUncond, err)) { report = rs.str() + "mnn uncond: " + err; return false; }
    if (embCond.size() != kEmbedElems || embUncond.size() != kEmbedElems) {
        report = rs.str() + "mnn output size mismatch"; return false;
    }
    rs << "[clip] cond=" << tokensCond.size() << "tok uncond=" << tokensUncond.size() << "tok\n";

    // VAE encode the input image.
    std::vector<float> imageLatent;
    std::string vaeEncReport;
    if (!vaeEncodeImage(bundle.vaeEncoderBin, params.rawRgbFp32, imageLatent, vaeEncReport)) {
        report = rs.str() + "stage encoding: " + vaeEncReport; return false;
    }
    rs << "[" << vaeEncReport << "]\n";

    // -------- Stage 5: diffusion (inpaint UNet). --------
    if (onStage) onStage("diffusing");
    std::vector<float> finalLatent;
    std::string diffReport;
    if (!runInpaintDiffusion(params.inpaintUnetBin,
                             embCond, embUncond,
                             imageLatent, mask64,
                             params.iterations, params.seed,
                             onStep, finalLatent, diffReport)) {
        report = rs.str() + "stage diffusing: " + diffReport; return false;
    }
    rs << "[" << diffReport << "]\n";

    // -------- Stage 6: VAE decode → PNG (reuse runDiffusion's helper). --------
    if (onStage) onStage("decoding");
    std::string vaeDecReport;
    if (!vaeDecodeToPng(params.bundleDir, finalLatent, pngOut, vaeDecReport)) {
        report = rs.str() + "stage decoding: " + vaeDecReport; return false;
    }
    rs << "[" << vaeDecReport << "]";

    auto t1 = clk::now();
    rs << "outfit-swap OK total="
       << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms";
    report = rs.str();
    __android_log_print(ANDROID_LOG_INFO, kTag, "%s", report.c_str());
    return true;
}

}  // namespace imagegen
