#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace imagegen {

// Outfit-swap pipeline entry point (PLAN-OUTFIT-SWAP.md). Single-call: input
// portrait photo + prompt + garment class bitfield → PNG with the masked
// garment region repainted by the SD 1.5 inpaint UNet.
//
// Pipeline (matches the plan diagram in §3):
//   1. SegFormer-B2-Clothes classmap from rawRgbFp32 (ImageNet-normalized inside).
//   2. mask_ops chain: argmax classmap → binary mask → dilate 5px → feather σ=2
//      → 8× block-mean downsample to 64×64 latent mask.
//   3. CLIP text encode (cond prompt + empty uncond) via the bundle's MNN model.
//   4. VAE encode the input photo → 4-ch image latent (1×4×64×64).
//   5. Build masked-image latent = image_latent ⊙ (1 - mask_latent_broadcast).
//   6. Inpaint UNet diffusion loop (DDIM, CFG, N steps). 9-channel input:
//      concat(z_t [4ch], mask [1ch], masked_image_latent [4ch]) along channels.
//      mask + masked-image latents are precomputed once; only z_t changes/step.
//   7. VAE decode the final latent → RGB → PNG (reuses diffusion::vaeDecodeToPng).
//
// Models orchestrated as load-run-unload to fit Hexagon context residency
// limits (PLAN §5): Seg → CLIP + VAE-enc → UNet → VAE-dec. Peak resident is
// the UNet (~1.5 GB w8a16).
//
// `onStage` (optional) is called once per stage transition with one of:
//   "segmenting" | "encoding" | "diffusing" | "decoding"
// `onStep` (optional) fires once per UNet step with (i, totalSteps).
struct OutfitSwapParams {
    // Caller-prepared 3×512×512 fp32 CHW row-major image, values in [0,1].
    std::vector<float> rawRgbFp32;

    // Bundle directory (xororz layout — provides CLIP + VAE enc/dec + tokenizer).
    std::string bundleDir;

    // AI Hub-compiled SegFormer-B2-Clothes context binary.
    std::string segformerBin;

    // AI Hub-compiled SD 1.5 inpaint UNet context binary (9-channel input).
    std::string inpaintUnetBin;

    // Free-text outfit description.
    std::string prompt;

    // Mask classes to repaint, as a bitfield over SegFormer's 18 ATR classes.
    // Use the kGarment* presets in mask_ops.hpp or build a custom bitmask.
    uint32_t selectedClasses;

    int iterations;
    uint64_t seed;
};

using StageCallback = std::function<void(const std::string& stageName)>;
using StepCallback  = std::function<void(int step, int totalSteps)>;

bool runOutfitSwap(const OutfitSwapParams& params,
                   StageCallback           onStage,
                   StepCallback            onStep,
                   std::vector<uint8_t>&   pngOut,
                   std::string&            report);

}  // namespace imagegen
