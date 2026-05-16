#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imagegen {

// Phase 0c+d entry point: run mattmdjaga/segformer_b2_clothes on an RGB image
// and produce a colored class-map visualization PNG.
//
// Inputs:
//   modelBinPath  Absolute path to the AI Hub-compiled QNN context binary
//                 (segformer_b2_clothes.bin). The binary expects a 1×3×512×512
//                 float32 input (already normalized to ImageNet stats), and
//                 produces 1×18×H×W logits where (H,W) is either 128×128 (HF
//                 default decoder head) or 512×512 (if the upsample is baked
//                 into the compile). Layout (NCHW vs NHWC) is detected at
//                 inspection time.
//   inputRgbFp32  Flat 3×512×512 float32 buffer, CHW row-major, ImageNet-
//                 normalized: x = (rgb01 - mean) / std with
//                 mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]. Caller is
//                 responsible for resize + normalization (done on the Kotlin
//                 side via Bitmap + pre-pass).
//   palettePngOut Filled with a complete PNG byte stream visualizing the
//                 argmaxed class map at the model's native output resolution.
//                 Each of the 18 ATR-style classes maps to a distinct color
//                 (see kPalette in segformer.cpp).
//   report        Multi-line diagnostic: graph signature, output dims, timing.
//                 Populated on success and failure both.
//
// Returns true on success; false with `report` populated on failure.
bool runSegformerMaskPng(const std::string&             modelBinPath,
                         const std::vector<float>&      inputRgbFp32,
                         std::vector<uint8_t>&          palettePngOut,
                         std::string&                   report);

// Outfit-swap Phase 3 entry point: load the model, run it, and return the
// argmaxed class map (uint8, H*W row-major) WITHOUT the visualization step.
// The output H and W are whatever the model emits (typically 128×128 for the
// stock decoder head; could be 512×512 if compiled with upsample baked in).
//
// `rawRgbFp32` is the *unnormalized* [0,1] RGB CHW image; the function applies
// ImageNet normalization internally. This is the canonical input shape we hand
// the entire outfit-swap pipeline (SegFormer + VAE share a starting buffer).
bool runSegformerClassmap(const std::string&             modelBinPath,
                          const std::vector<float>&      rawRgbFp32,
                          std::vector<uint8_t>&          classmapOut,
                          int&                           outH,
                          int&                           outW,
                          std::string&                   report);

// Expected element count of `inputRgbFp32` — exposed for the JNI shim's input
// validation so the size mismatch errors are caught before crossing the bridge.
inline constexpr int kSegformerInputC = 3;
inline constexpr int kSegformerInputH = 512;
inline constexpr int kSegformerInputW = 512;
inline constexpr int kSegformerInputElems =
    kSegformerInputC * kSegformerInputH * kSegformerInputW;
inline constexpr int kSegformerNumClasses = 18;

}  // namespace imagegen
