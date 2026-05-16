#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imagegen {

// Outfit-swap mask post-processing. The full chain is:
//   SegFormer logits (1×18×H×W)
//     → argmax classmap (uint8, 1×H×W)            // segformer.cpp does this
//     → binary mask (uint8 0/255, 1×H_full×W_full) // classmapToBinaryMask
//     → dilate 5 px (uint8 0/255)                  // dilateMask5px
//     → Gaussian feather σ=2 (float 0..1)          // featherMaskGaussian
//     → 8× downsample to latent (float 0..1)       // downsampleMaskNearest8x
//     → broadcast over 4-ch latent for masked-image
//                                                  // applyMaskToLatent
//
// The "selected classes" bitfield maps directly to SegFormer's 18-class ATR
// scheme; see PLAN-OUTFIT-SWAP.md Appendix B. Bit i = class i.

inline constexpr uint32_t bit(int i) { return 1u << i; }

inline constexpr uint32_t kSegClassBackground   = bit(0);
inline constexpr uint32_t kSegClassHat          = bit(1);
inline constexpr uint32_t kSegClassHair         = bit(2);
inline constexpr uint32_t kSegClassSunglasses   = bit(3);
inline constexpr uint32_t kSegClassUpperClothes = bit(4);
inline constexpr uint32_t kSegClassSkirt        = bit(5);
inline constexpr uint32_t kSegClassPants        = bit(6);
inline constexpr uint32_t kSegClassDress        = bit(7);
inline constexpr uint32_t kSegClassBelt         = bit(8);
inline constexpr uint32_t kSegClassLeftShoe     = bit(9);
inline constexpr uint32_t kSegClassRightShoe    = bit(10);
inline constexpr uint32_t kSegClassFace         = bit(11);
inline constexpr uint32_t kSegClassLeftLeg      = bit(12);
inline constexpr uint32_t kSegClassRightLeg     = bit(13);
inline constexpr uint32_t kSegClassLeftArm      = bit(14);
inline constexpr uint32_t kSegClassRightArm     = bit(15);
inline constexpr uint32_t kSegClassBag          = bit(16);
inline constexpr uint32_t kSegClassScarf        = bit(17);

// Garment-selector presets matching the widget chip row.
inline constexpr uint32_t kGarmentTop = kSegClassUpperClothes;
inline constexpr uint32_t kGarmentBottom = kSegClassSkirt | kSegClassPants;
inline constexpr uint32_t kGarmentDress = kSegClassDress;
inline constexpr uint32_t kGarmentFullBody =
    kSegClassUpperClothes | kSegClassSkirt | kSegClassPants | kSegClassDress;
inline constexpr uint32_t kGarmentAuto =
    kSegClassHat | kSegClassUpperClothes | kSegClassSkirt | kSegClassPants |
    kSegClassDress | kSegClassBelt | kSegClassScarf;

// SegFormer's argmax output ([H,W] uint8 class IDs) → binary mask (1=in mask).
// Output is `255` where the class is in `selectedClasses`, `0` elsewhere.
// Output buffer is resized to H*W; H,W match input.
void classmapToBinaryMask(const std::vector<uint8_t>& classmap,
                          int H, int W,
                          uint32_t selectedClasses,
                          std::vector<uint8_t>& maskOut);

// 5-pixel dilation with a square structuring element. Cheap separable pass:
// for each output pixel, take the max of a (2*radius+1)-wide window along x,
// then along y. Operates in place if `out` aliases `in` is NOT supported —
// caller must pass disjoint buffers. Inputs are 0/255; output is 0/255.
void dilateMask(const std::vector<uint8_t>& in,
                int H, int W,
                int radiusPixels,
                std::vector<uint8_t>& out);

// Convenience for the default 5-pixel dilation.
inline void dilateMask5px(const std::vector<uint8_t>& in, int H, int W,
                          std::vector<uint8_t>& out) {
    dilateMask(in, H, W, /*radius=*/5, out);
}

// Gaussian feather of the binary mask into a float mask in [0,1]. Separable
// pass with a small kernel sized from `sigma` (truncated at ±3σ). Output
// element count is H*W; values are normalized.
void featherMaskGaussian(const std::vector<uint8_t>& in,
                         int H, int W,
                         float sigma,
                         std::vector<float>& out);

// Downsample an H×W float mask to (H/factor)×(W/factor) by simple block-mean
// pooling (NOT nearest-neighbor: averaging gives a soft latent mask that the
// inpaint UNet expects). Default factor = 8 (512→64 for SD 1.5 latents).
void downsampleMaskMean(const std::vector<float>& in,
                        int H, int W,
                        int factor,
                        std::vector<float>& out);

// 4-channel "masked image latent": for each latent element, copy the original
// image latent where the mask is 0, and zero it where the mask is 1.
// `imageLatent` is flat 4*H*W (CHW), `maskLatent` is flat H*W. Output overwrites
// `maskedOut` and matches imageLatent's shape.
//
// SD 1.5 inpaint convention: `masked_image_latents = image_latents * (1 - mask)`.
// (Confirmed against diffusers pipeline_stable_diffusion_inpaint.py.)
void buildMaskedImageLatent(const std::vector<float>& imageLatent,
                            const std::vector<float>& maskLatent,
                            int H, int W,
                            std::vector<float>& maskedOut);

}  // namespace imagegen
