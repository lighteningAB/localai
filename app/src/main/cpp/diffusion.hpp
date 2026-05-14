#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imagegen {

// Phase 6 deliverable: drive the full SD 1.5 diffusion loop with classifier-
// free guidance using the QNN-loaded UNet + MNN-loaded text encoder + the
// pure-C++ DPM-Solver++ scheduler.
//
// Returns the final latent tensor (fp32, length 1*4*64*64 = 16384) on success.
// On failure returns an empty vector and populates `report` with a diagnostic.
//
// `report` is also populated on success with a multi-line summary (timings,
// UNet graph signature, latent-stat sanity check) — used by the boot probe
// in LocalAiApp.kt for verifiable end-to-end success without yet doing the
// Phase 7 VAE decode.
std::vector<float> runDiffusion(const std::string& bundleDir,
                                const std::string& prompt,
                                int                iters,
                                uint64_t           seed,
                                std::string&       report);

// Phase 7: VAE-decode a final UNet latent and PNG-encode the resulting RGB
// image. Loads `vae_decoder.bin` from the bundle into a fresh QnnSession,
// scales latents by 1/0.18215 (SD 1.5 convention), runs the decoder, denorms
// the output to uint8 RGB (HWC interleaved), and writes a PNG byte stream into
// `pngOut`. Layout (NCHW vs NHWC) is detected from the graph metadata.
//
// Returns false on failure with a diagnostic in `report`. On success `report`
// holds a single-line summary (output dims, layout, png byte count).
bool vaeDecodeToPng(const std::string&        bundleDir,
                    const std::vector<float>& latents,
                    std::vector<uint8_t>&     pngOut,
                    std::string&              report);

// One-shot: runDiffusion + vaeDecodeToPng. Convenience for the JNI surface.
// On failure returns false and `report` contains a diagnostic; on success
// `pngOut` is the encoded PNG and `report` aggregates both step reports.
bool runDiffusionToPng(const std::string&    bundleDir,
                       const std::string&    prompt,
                       int                   iters,
                       uint64_t              seed,
                       std::vector<uint8_t>& pngOut,
                       std::string&          report);

}  // namespace imagegen
