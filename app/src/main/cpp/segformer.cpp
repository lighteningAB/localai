// Phase 0c+d: run mattmdjaga/segformer_b2_clothes via QNN and emit a colored
// class-map PNG. Self-contained — duplicates a few small tensor helpers from
// diffusion.cpp rather than introducing a shared header for them. Once Phase 3
// wires this into the outfit-swap pipeline, the helpers move to mask_ops.{hpp,cpp}.

#include "segformer.hpp"

#include <android/log.h>

#include <chrono>
#include <cstring>
#include <sstream>

#include "png_encode.hpp"
#include "qnn_session.hpp"

namespace imagegen {

namespace {

constexpr const char* kTag = "segformer";

// QNN_DATATYPE_* numeric values (mirror QnnTypes.h). Same set as diffusion.cpp.
constexpr uint32_t kDtFloat16 = 0x0216;
constexpr uint32_t kDtFloat32 = 0x0232;
constexpr uint32_t kDtUfp16   = 0x0416;  // UFIXED_POINT_16 — w8a16 quant boundary

// 18-color palette for the ATR/iMaterialist classes mattmdjaga's model emits.
// Class IDs (from the model card):
//   0 Background, 1 Hat, 2 Hair, 3 Sunglasses, 4 Upper-clothes, 5 Skirt,
//   6 Pants, 7 Dress, 8 Belt, 9 Left-shoe, 10 Right-shoe, 11 Face, 12 Left-leg,
//   13 Right-leg, 14 Left-arm, 15 Right-arm, 16 Bag, 17 Scarf.
// Colors picked for high contrast at small thumbnail sizes.
constexpr uint8_t kPalette[kSegformerNumClasses][3] = {
    {  0,   0,   0},  // 0  Background — black
    {128,   0,   0},  // 1  Hat        — dark red
    {255, 200,   0},  // 2  Hair       — gold
    {  0, 200, 255},  // 3  Sunglasses — cyan
    {255,  50,  50},  // 4  Upper-clothes — bright red (the v1 hero class)
    {  0, 200,   0},  // 5  Skirt      — green
    { 50,  50, 220},  // 6  Pants      — blue
    {255, 100, 200},  // 7  Dress      — pink
    {200, 200,   0},  // 8  Belt       — yellow
    {120,  60,  20},  // 9  Left-shoe  — brown
    {180,  90,  30},  // 10 Right-shoe — light brown
    {255, 220, 180},  // 11 Face       — skin
    {180, 100, 100},  // 12 Left-leg   — dusty red
    {200, 120, 120},  // 13 Right-leg  — dusty red lighter
    {100, 180, 200},  // 14 Left-arm   — teal
    {130, 200, 220},  // 15 Right-arm  — teal lighter
    {150,  80, 200},  // 16 Bag        — purple
    {255, 180,  60},  // 17 Scarf      — orange
};

// IEEE-754 half ↔ float32. Lifted verbatim from diffusion.cpp; intentionally
// duplicated (one of two tiny call sites; not worth a shared header yet).
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
        if ((mant >> (shift - 1)) & 1u) half += 1;
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | (0x1Fu << 10));
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

// Pack a flat float32 vector into the on-the-wire buffer the QNN graph expects.
// Handles fp32 and ufp16 (w8a16 quant boundary tensors). All other dtypes are
// rejected — fp16 would be reachable too but Hub w8a16 doesn't produce it on
// input tensors in practice.
bool packFp32Input(const std::vector<float>& src,
                   const QnnTensorInfo&      info,
                   std::vector<uint8_t>&     bytes,
                   std::string&              error) {
    const std::size_t elems = src.size();
    const std::size_t outBytes = tensorByteSize(info);
    bytes.assign(outBytes, 0);
    const uint32_t dt = info.dataType;
    if (dt == kDtFloat32) {
        if (outBytes != elems * sizeof(float)) {
            error = "packFp32Input: fp32 tensor size mismatch";
            return false;
        }
        std::memcpy(bytes.data(), src.data(), outBytes);
        return true;
    }
    if (dt == kDtFloat16) {
        if (outBytes != elems * 2) {
            error = "packFp32Input: fp16 tensor size mismatch";
            return false;
        }
        auto* dst = reinterpret_cast<uint16_t*>(bytes.data());
        for (std::size_t i = 0; i < elems; ++i) dst[i] = floatToHalf(src[i]);
        return true;
    }
    if (dt == kDtUfp16) {
        if (info.quantScale == 0.0f) {
            error = "packFp32Input: ufp16 tensor missing scale";
            return false;
        }
        if (outBytes != elems * 2) {
            error = "packFp32Input: ufp16 tensor size mismatch";
            return false;
        }
        const float inv = 1.0f / info.quantScale;
        auto* dst = reinterpret_cast<uint16_t*>(bytes.data());
        for (std::size_t i = 0; i < elems; ++i) {
            int32_t q = static_cast<int32_t>(std::nearbyint(src[i] * inv)) - info.quantOffset;
            if (q < 0) q = 0;
            if (q > 0xFFFF) q = 0xFFFF;
            dst[i] = static_cast<uint16_t>(q);
        }
        return true;
    }
    error = "packFp32Input: unsupported dtype 0x" + std::to_string(dt);
    return false;
}

// Unpack QNN output bytes into a flat fp32 vector. Mirror of packFp32Input.
bool unpackToFp32(const std::vector<uint8_t>& src,
                  const QnnTensorInfo&        info,
                  std::size_t                 expectedElems,
                  std::vector<float>&         out,
                  std::string&                error) {
    out.assign(expectedElems, 0.0f);
    const uint32_t dt = info.dataType;
    if (dt == kDtFloat32) {
        if (src.size() < expectedElems * sizeof(float)) {
            error = "unpackToFp32: undersized fp32 buffer";
            return false;
        }
        std::memcpy(out.data(), src.data(), expectedElems * sizeof(float));
        return true;
    }
    if (dt == kDtFloat16) {
        if (src.size() < expectedElems * 2) {
            error = "unpackToFp32: undersized fp16 buffer";
            return false;
        }
        const auto* p = reinterpret_cast<const uint16_t*>(src.data());
        for (std::size_t i = 0; i < expectedElems; ++i) out[i] = halfToFloat(p[i]);
        return true;
    }
    if (dt == kDtUfp16) {
        if (src.size() < expectedElems * 2) {
            error = "unpackToFp32: undersized ufp16 buffer";
            return false;
        }
        if (info.quantScale == 0.0f) {
            error = "unpackToFp32: ufp16 tensor missing scale";
            return false;
        }
        const auto* p = reinterpret_cast<const uint16_t*>(src.data());
        const float scale = info.quantScale;
        const int32_t off = info.quantOffset;
        for (std::size_t i = 0; i < expectedElems; ++i) {
            out[i] = (static_cast<float>(p[i]) + static_cast<float>(off)) * scale;
        }
        return true;
    }
    error = "unpackToFp32: unsupported dtype 0x" + std::to_string(dt);
    return false;
}

// Pick H,W and layout (NCHW vs NHWC) given the expected channel count. Mirrors
// classifyLayout4D in diffusion.cpp.
bool classify4D(const std::vector<uint32_t>& dims,
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

// Argmax across the class axis. `logits` is flat fp32 of size C*H*W (or H*W*C
// for NHWC). Output is a uint8 class map of size H*W, in row-major order.
void argmaxToClassmap(const std::vector<float>& logits,
                      int C, int H, int W, bool isNhwc,
                      std::vector<uint8_t>& classmap) {
    classmap.assign(static_cast<std::size_t>(H) * W, 0);
    if (isNhwc) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const std::size_t base = (static_cast<std::size_t>(h) * W + w) * C;
                float best = logits[base];
                int   arg  = 0;
                for (int c = 1; c < C; ++c) {
                    const float v = logits[base + c];
                    if (v > best) { best = v; arg = c; }
                }
                classmap[static_cast<std::size_t>(h) * W + w] = static_cast<uint8_t>(arg);
            }
        }
    } else {
        // NCHW: stride over plane size, then per pixel.
        const std::size_t plane = static_cast<std::size_t>(H) * W;
        for (std::size_t i = 0; i < plane; ++i) {
            float best = logits[i];
            int   arg  = 0;
            for (int c = 1; c < C; ++c) {
                const float v = logits[c * plane + i];
                if (v > best) { best = v; arg = c; }
            }
            classmap[i] = static_cast<uint8_t>(arg);
        }
    }
}

// Apply the fixed 18-color palette to a class map. Unknown class IDs (>= 18)
// pass through as magenta so they're visually obvious.
void colorizeClassmap(const std::vector<uint8_t>& classmap,
                      int H, int W,
                      std::vector<uint8_t>& rgbOut) {
    rgbOut.assign(static_cast<std::size_t>(H) * W * 3, 0);
    for (std::size_t i = 0; i < classmap.size(); ++i) {
        const uint8_t c = classmap[i];
        const uint8_t* color =
            (c < kSegformerNumClasses) ? kPalette[c]
                                       : reinterpret_cast<const uint8_t*>("\xFF\x00\xFF");
        rgbOut[i * 3 + 0] = color[0];
        rgbOut[i * 3 + 1] = color[1];
        rgbOut[i * 3 + 2] = color[2];
    }
}

// ImageNet stats (the preprocessing mattmdjaga's model expects).
constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};

// Convert a raw [0,1] RGB CHW buffer into the ImageNet-normalized buffer the
// model expects. New allocation; caller keeps the original.
void imagenetNormalizeCHW(const std::vector<float>& raw,
                          std::vector<float>&       normalized) {
    normalized.resize(raw.size());
    const std::size_t plane =
        static_cast<std::size_t>(kSegformerInputH) * kSegformerInputW;
    for (int c = 0; c < kSegformerInputC; ++c) {
        const float m = kImageNetMean[c];
        const float s = kImageNetStd[c];
        for (std::size_t i = 0; i < plane; ++i) {
            const std::size_t k = static_cast<std::size_t>(c) * plane + i;
            normalized[k] = (raw[k] - m) / s;
        }
    }
}

// Core inference path. Takes already-ImageNet-normalized CHW input. Fills
// `classmap` (uint8 H*W argmax) plus `outH`/`outW`. Used by both the PNG
// visualization entry point and the outfit-swap pipeline.
bool runSegformerCore(const std::string&        modelBinPath,
                      const std::vector<float>& normalizedRgb,
                      std::vector<uint8_t>&     classmap,
                      int&                      outHOut,
                      int&                      outWOut,
                      bool&                     outIsNhwc,
                      std::ostringstream&       rs) {
    if (static_cast<int>(normalizedRgb.size()) != kSegformerInputElems) {
        rs << "seg input size " << normalizedRgb.size()
           << " != expected " << kSegformerInputElems;
        return false;
    }
    QnnSession qnn;
    std::string err;
    if (!qnn.initialize(err))                  { rs << "seg init: " << err; return false; }
    if (!qnn.inspectBinary(modelBinPath, err)) { rs << "seg inspect: " << err; return false; }
    if (!qnn.instantiate(err))                 { rs << "seg instantiate: " << err; return false; }

    if (qnn.graphs().empty()) { rs << "seg binary has no graphs"; return false; }
    const auto& g = qnn.graphs()[0];
    if (g.inputs.size() != 1 || g.outputs.size() != 1) {
        rs << "seg expects 1 in/1 out, got in=" << g.inputs.size()
           << " out=" << g.outputs.size();
        return false;
    }
    const auto& inInfo  = g.inputs[0];
    const auto& outInfo = g.outputs[0];

    int inH = 0, inW = 0; bool inIsNhwc = false;
    if (!classify4D(inInfo.dims, 3u, inH, inW, inIsNhwc)) {
        rs << "seg input dims not classifiable: rank=" << inInfo.dims.size();
        return false;
    }
    if (inH != kSegformerInputH || inW != kSegformerInputW) {
        rs << "seg input shape mismatch H=" << inH << " W=" << inW;
        return false;
    }

    std::vector<float> packSrc;
    if (inIsNhwc) {
        packSrc.resize(normalizedRgb.size());
        const int C = kSegformerInputC;
        const int H = kSegformerInputH;
        const int W = kSegformerInputW;
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    packSrc[(static_cast<std::size_t>(h) * W + w) * C + c] =
                        normalizedRgb[(static_cast<std::size_t>(c) * H + h) * W + w];
                }
            }
        }
    }
    const std::vector<float>& packView = inIsNhwc ? packSrc : normalizedRgb;

    std::vector<uint8_t> inBytes;
    if (!packFp32Input(packView, inInfo, inBytes, err)) {
        rs << "seg pack: " << err;
        return false;
    }

    int outH = 0, outW = 0;
    if (!classify4D(outInfo.dims, kSegformerNumClasses, outH, outW, outIsNhwc)) {
        rs << "seg output dims not classifiable (rank=" << outInfo.dims.size() << ")";
        return false;
    }
    const std::size_t outElems =
        static_cast<std::size_t>(kSegformerNumClasses) * outH * outW;

    std::vector<std::vector<uint8_t>> inBufs(1);
    inBufs[0] = std::move(inBytes);
    std::vector<std::vector<uint8_t>> outBufs(1);
    outBufs[0].assign(tensorByteSize(outInfo), 0);

    if (!qnn.execute(0, inBufs, outBufs, err)) {
        rs << "seg execute: " << err;
        return false;
    }

    std::vector<float> logits;
    if (!unpackToFp32(outBufs[0], outInfo, outElems, logits, err)) {
        rs << "seg unpack: " << err;
        return false;
    }
    argmaxToClassmap(logits, kSegformerNumClasses, outH, outW, outIsNhwc, classmap);

    outHOut = outH;
    outWOut = outW;
    rs << "seg in=" << (inIsNhwc ? "NHWC" : "NCHW")
       << " out=" << (outIsNhwc ? "NHWC" : "NCHW")
       << " " << outH << "x" << outW;
    return true;
}

}  // namespace

bool runSegformerClassmap(const std::string&        modelBinPath,
                          const std::vector<float>& rawRgbFp32,
                          std::vector<uint8_t>&     classmapOut,
                          int&                      outH,
                          int&                      outW,
                          std::string&              report) {
    std::ostringstream rs;
    std::vector<float> normalized;
    imagenetNormalizeCHW(rawRgbFp32, normalized);
    bool outIsNhwc = false;
    const bool ok =
        runSegformerCore(modelBinPath, normalized, classmapOut, outH, outW, outIsNhwc, rs);
    report = rs.str();
    if (ok) {
        __android_log_print(ANDROID_LOG_INFO, kTag, "classmap %s", report.c_str());
    } else {
        __android_log_print(ANDROID_LOG_WARN, kTag, "classmap FAIL %s", report.c_str());
    }
    return ok;
}

bool runSegformerMaskPng(const std::string&        modelBinPath,
                         const std::vector<float>& inputRgbFp32,
                         std::vector<uint8_t>&     palettePngOut,
                         std::string&              report) {
    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    std::ostringstream rs;

    if (static_cast<int>(inputRgbFp32.size()) != kSegformerInputElems) {
        report = "input size " + std::to_string(inputRgbFp32.size()) +
                 " != expected " + std::to_string(kSegformerInputElems);
        return false;
    }

    QnnSession qnn;
    std::string err;
    if (!qnn.initialize(err))                       { report = "seg init: " + err; return false; }
    if (!qnn.inspectBinary(modelBinPath, err))      { report = "seg inspect: " + err; return false; }
    if (!qnn.instantiate(err))                      { report = "seg instantiate: " + err; return false; }
    auto t_loaded = clk::now();

    if (qnn.graphs().empty()) { report = "seg binary has no graphs"; return false; }
    const auto& g = qnn.graphs()[0];
    if (g.inputs.size() != 1 || g.outputs.size() != 1) {
        report = "seg expects 1 in/1 out, got in=" + std::to_string(g.inputs.size()) +
                 " out=" + std::to_string(g.outputs.size());
        return false;
    }
    const auto& inInfo  = g.inputs[0];
    const auto& outInfo = g.outputs[0];

    // Input must be a 3-channel 512×512 image tensor.
    int inH = 0, inW = 0; bool inIsNhwc = false;
    if (!classify4D(inInfo.dims, 3u, inH, inW, inIsNhwc)) {
        report = "seg input dims not classifiable as 3-channel 4D: '" +
                 inInfo.name + "' rank=" + std::to_string(inInfo.dims.size());
        return false;
    }
    if (inH != kSegformerInputH || inW != kSegformerInputW) {
        report = "seg input shape mismatch: got H=" + std::to_string(inH) +
                 " W=" + std::to_string(inW) +
                 " expected " + std::to_string(kSegformerInputH) + "x" +
                 std::to_string(kSegformerInputW);
        return false;
    }

    // Input arrives CHW (Kotlin promised). If the graph expects NHWC we need
    // to transpose before packing.
    std::vector<float> packSrc;
    if (inIsNhwc) {
        packSrc.resize(inputRgbFp32.size());
        const int C = kSegformerInputC;
        const int H = kSegformerInputH;
        const int W = kSegformerInputW;
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    packSrc[(static_cast<std::size_t>(h) * W + w) * C + c] =
                        inputRgbFp32[(static_cast<std::size_t>(c) * H + h) * W + w];
                }
            }
        }
    }
    const std::vector<float>& packView = inIsNhwc ? packSrc : inputRgbFp32;

    std::vector<uint8_t> inBytes;
    if (!packFp32Input(packView, inInfo, inBytes, err)) {
        report = "seg pack input: " + err;
        return false;
    }

    // Output must be 18-channel logits, square.
    int outH = 0, outW = 0; bool outIsNhwc = false;
    if (!classify4D(outInfo.dims, kSegformerNumClasses, outH, outW, outIsNhwc)) {
        report = "seg output dims not classifiable as " +
                 std::to_string(kSegformerNumClasses) + "-channel 4D: '" +
                 outInfo.name + "' rank=" + std::to_string(outInfo.dims.size());
        return false;
    }
    const std::size_t outElems =
        static_cast<std::size_t>(kSegformerNumClasses) * outH * outW;

    std::vector<std::vector<uint8_t>> inBufs(1);
    inBufs[0] = std::move(inBytes);
    std::vector<std::vector<uint8_t>> outBufs(1);
    outBufs[0].assign(tensorByteSize(outInfo), 0);

    auto t_exec_start = clk::now();
    if (!qnn.execute(0, inBufs, outBufs, err)) {
        report = "seg execute: " + err;
        return false;
    }
    auto t_exec_end = clk::now();

    std::vector<float> logits;
    if (!unpackToFp32(outBufs[0], outInfo, outElems, logits, err)) {
        report = "seg unpack: " + err;
        return false;
    }

    std::vector<uint8_t> classmap;
    argmaxToClassmap(logits, kSegformerNumClasses, outH, outW, outIsNhwc, classmap);

    // Quick class histogram for sanity (helps diagnose dead-class collapses).
    int hist[kSegformerNumClasses] = {};
    for (uint8_t c : classmap) {
        if (c < kSegformerNumClasses) ++hist[c];
    }

    std::vector<uint8_t> rgb;
    colorizeClassmap(classmap, outH, outW, rgb);

    if (!encodeRgbToPng(outW, outH, rgb, palettePngOut, err)) {
        report = "seg png encode: " + err;
        return false;
    }

    auto t_end = clk::now();
    rs << "seg in=" << (inIsNhwc ? "NHWC" : "NCHW") << " 512x512 fp32→"
       << (inInfo.dataType == kDtFloat32 ? "fp32" :
           inInfo.dataType == kDtFloat16 ? "fp16" :
           inInfo.dataType == kDtUfp16   ? "ufp16(s=" + std::to_string(inInfo.quantScale) + ")"
                                         : "dt=" + std::to_string(inInfo.dataType))
       << " out=" << (outIsNhwc ? "NHWC" : "NCHW")
       << " " << outH << "x" << outW
       << " classes=" << kSegformerNumClasses
       << " pngBytes=" << palettePngOut.size();
    rs << " hist=[";
    for (int i = 0; i < kSegformerNumClasses; ++i) {
        if (i) rs << ",";
        rs << hist[i];
    }
    rs << "]";
    rs << " timing: load=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_loaded - t_start).count() << "ms"
       <<         " exec=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_exec_end - t_exec_start).count() << "ms"
       <<        " total=" << std::chrono::duration_cast<std::chrono::milliseconds>(
              t_end - t_start).count() << "ms";

    report = rs.str();
    __android_log_print(ANDROID_LOG_INFO, kTag, "%s", report.c_str());
    return true;
}

}  // namespace imagegen
