#pragma once

#include <string>
#include <vector>

namespace imagegen {

// Manifest of files in an xororz SD-QNN bundle, mapped to absolute paths on the
// device. The bundles are plain zip archives — the user extracts on the host
// and pushes the directory via scripts/push-diffusion-bundle.sh. No in-process
// zip extraction; this struct is the "what's where" lookup.
//
// xororz bundle layout (verified against AbsoluteReality_qnn2.28_8gen2.zip):
//   tokenizer.json     HF tokenizer JSON (CLIP BPE) — tokenizers-cpp consumes directly
//   clip_v2.mnn        Text encoder as Alibaba MNN model (CPU / OpenCL on GPU)
//   pos_emb.bin        CLIP position embeddings
//   token_emb.bin      CLIP token embeddings (fp16)
//   vae_encoder.bin    VAE encoder QNN context binary
//   vae_decoder.bin    VAE decoder QNN context binary
//   unet.bin           UNet QNN context binary (largest file, ~840 MB for 8gen2)
//   *.patch            Resolution-specific UNet overlays (512x768, 768.patch, etc.)
//
// Note: the text encoder is MNN, not QNN — xororz uses QNN only for UNet + VAE.
// This deviates from IMAGE-GEN-PLAN.md §2 which framed MNN as a CPU fallback.
struct Bundle {
    std::string root;            // absolute path of the extracted bundle directory
    std::string tokenizerJson;
    std::string clipMnn;
    std::string posEmbBin;
    std::string tokenEmbBin;
    std::string vaeEncoderBin;
    std::string vaeDecoderBin;
    std::string unetBin;
    std::vector<std::string> patchFiles;  // optional resolution overlays
};

// Load and validate a bundle directory. Returns true if all required files exist;
// fills `out` with absolute paths. On failure returns false and populates `error`
// with a human-readable reason.
//
// `rootDir` should be the absolute path of the extracted bundle directory (the
// one that directly contains tokenizer.json, unet.bin, etc.). xororz zips include
// a top-level "output_512/qnn_models_8gen2/" wrapper directory; the push script
// strips that, so on-device the wrapper is gone.
bool loadBundle(const std::string& rootDir, Bundle& out, std::string& error);

}  // namespace imagegen
