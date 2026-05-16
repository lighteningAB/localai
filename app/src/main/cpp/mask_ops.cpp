// Phase 3 outfit-swap mask post-processing. Pure CPU helpers — no QNN, no MNN.
// Easy to unit-test on host once we wire a test target; for now exercised
// end-to-end via the outfit-swap pipeline.

#include "mask_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace imagegen {

void classmapToBinaryMask(const std::vector<uint8_t>& classmap,
                          int H, int W,
                          uint32_t selectedClasses,
                          std::vector<uint8_t>& maskOut) {
    const std::size_t n = static_cast<std::size_t>(H) * W;
    maskOut.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t c = classmap[i];
        if (c < 32 && (selectedClasses & (1u << c)) != 0) {
            maskOut[i] = 255;
        }
    }
}

// 1D max filter, window radius `r`, in-place along columns of length `len`
// stepping by `stride` elements. Helper for separable dilation.
namespace {

void max1dRow(const uint8_t* src, uint8_t* dst, int len, int r) {
    // Naive O(len * r). For r=5 on H=W=512 this is ~5 ns/pixel; the whole
    // dilate pass is well under 10 ms. A monotonic-deque O(len) is a future
    // optimization if the profile says it matters.
    for (int i = 0; i < len; ++i) {
        const int lo = std::max(0, i - r);
        const int hi = std::min(len - 1, i + r);
        uint8_t m = 0;
        for (int j = lo; j <= hi; ++j) {
            if (src[j] > m) m = src[j];
        }
        dst[i] = m;
    }
}

void max1dCol(const uint8_t* src, uint8_t* dst, int H, int W, int r) {
    std::vector<uint8_t> colSrc(static_cast<std::size_t>(H));
    std::vector<uint8_t> colDst(static_cast<std::size_t>(H));
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) colSrc[y] = src[y * W + x];
        max1dRow(colSrc.data(), colDst.data(), H, r);
        for (int y = 0; y < H; ++y) dst[y * W + x] = colDst[y];
    }
}

void gauss1dRow(const float* src, float* dst, int len,
                const std::vector<float>& kernel, int r) {
    for (int i = 0; i < len; ++i) {
        float acc = 0.0f;
        for (int k = -r; k <= r; ++k) {
            int j = i + k;
            if (j < 0) j = 0;
            else if (j >= len) j = len - 1;
            acc += src[j] * kernel[static_cast<std::size_t>(k + r)];
        }
        dst[i] = acc;
    }
}

void gauss1dCol(const float* src, float* dst, int H, int W,
                const std::vector<float>& kernel, int r) {
    std::vector<float> colSrc(static_cast<std::size_t>(H));
    std::vector<float> colDst(static_cast<std::size_t>(H));
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) colSrc[y] = src[y * W + x];
        gauss1dRow(colSrc.data(), colDst.data(), H, kernel, r);
        for (int y = 0; y < H; ++y) dst[y * W + x] = colDst[y];
    }
}

}  // namespace

void dilateMask(const std::vector<uint8_t>& in,
                int H, int W,
                int radiusPixels,
                std::vector<uint8_t>& out) {
    const std::size_t n = static_cast<std::size_t>(H) * W;
    if (radiusPixels <= 0) {
        out = in;
        return;
    }
    std::vector<uint8_t> tmp(n);
    // Row pass.
    for (int y = 0; y < H; ++y) {
        max1dRow(in.data() + static_cast<std::size_t>(y) * W,
                 tmp.data() + static_cast<std::size_t>(y) * W,
                 W, radiusPixels);
    }
    // Column pass.
    out.assign(n, 0);
    max1dCol(tmp.data(), out.data(), H, W, radiusPixels);
}

void featherMaskGaussian(const std::vector<uint8_t>& in,
                         int H, int W,
                         float sigma,
                         std::vector<float>& out) {
    const std::size_t n = static_cast<std::size_t>(H) * W;
    if (sigma <= 0.0f) {
        out.assign(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = in[i] != 0 ? 1.0f : 0.0f;
        }
        return;
    }
    // Truncate the kernel at ±3σ; build a normalized 1D Gaussian.
    const int r = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> k(static_cast<std::size_t>(2 * r + 1));
    float kSum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float v = std::exp(-(i * i) / (2.0f * sigma * sigma));
        k[static_cast<std::size_t>(i + r)] = v;
        kSum += v;
    }
    for (float& v : k) v /= kSum;

    // Promote uint8 → float in [0,1]; then row + column passes.
    std::vector<float> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = in[i] != 0 ? 1.0f : 0.0f;
    }
    std::vector<float> tmp(n);
    for (int y = 0; y < H; ++y) {
        gauss1dRow(src.data() + static_cast<std::size_t>(y) * W,
                   tmp.data() + static_cast<std::size_t>(y) * W,
                   W, k, r);
    }
    out.assign(n, 0.0f);
    gauss1dCol(tmp.data(), out.data(), H, W, k, r);
    // Clamp [0,1] — the Gaussian is normalized so this is mostly defensive.
    for (float& v : out) {
        if (v < 0.0f) v = 0.0f;
        else if (v > 1.0f) v = 1.0f;
    }
}

void downsampleMaskMean(const std::vector<float>& in,
                        int H, int W,
                        int factor,
                        std::vector<float>& out) {
    const int outH = H / factor;
    const int outW = W / factor;
    out.assign(static_cast<std::size_t>(outH) * outW, 0.0f);
    const float inv = 1.0f / (factor * factor);
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            float acc = 0.0f;
            for (int dy = 0; dy < factor; ++dy) {
                const int sy = y * factor + dy;
                for (int dx = 0; dx < factor; ++dx) {
                    const int sx = x * factor + dx;
                    acc += in[static_cast<std::size_t>(sy) * W + sx];
                }
            }
            out[static_cast<std::size_t>(y) * outW + x] = acc * inv;
        }
    }
}

void buildMaskedImageLatent(const std::vector<float>& imageLatent,
                            const std::vector<float>& maskLatent,
                            int H, int W,
                            std::vector<float>& maskedOut) {
    constexpr int C = 4;
    const std::size_t plane = static_cast<std::size_t>(H) * W;
    maskedOut.assign(static_cast<std::size_t>(C) * plane, 0.0f);
    for (int c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < plane; ++i) {
            const float keep = 1.0f - maskLatent[i];
            maskedOut[c * plane + i] = imageLatent[c * plane + i] * keep;
        }
    }
}

}  // namespace imagegen
