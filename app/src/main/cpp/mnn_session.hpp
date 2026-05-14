#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imagegen {

struct MnnTensorInfo {
    std::string      name;
    int              elementBits = 0;   // 8 / 16 / 32 / 64
    bool             isFloat     = false;
    std::vector<int> dims;
};

// MNN-backed text encoder session. Loads clip_v2.mnn, runs forward on int32
// token IDs, returns the embedding output.
//
// Phase 4b deliverable: tokenize → encode → log first 8 output values on device.
// Phase 6 will reuse this session per generation (init is expensive).
class MnnSession {
public:
    MnnSession();
    ~MnnSession();

    MnnSession(const MnnSession&)            = delete;
    MnnSession& operator=(const MnnSession&) = delete;

    // Load `.mnn` file. preferOpenCL=true uses Adreno GPU; falls back to CPU
    // if the OpenCL backend is unavailable.
    bool load(const std::string& mnnPath, bool preferOpenCL, std::string& error);

    // Run forward pass with a pre-shaped fp32 input. The buffer length must
    // match the product of the model's input tensor dims. (For xororz's
    // clip_v2.mnn this is 1*77*768 = 59136.)
    bool runForward(const std::vector<float>& input,
                    std::vector<float>&       output,
                    std::string&              error);

    const std::vector<MnnTensorInfo>& inputs()  const;
    const std::vector<MnnTensorInfo>& outputs() const;
    std::string backendName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One-shot inspect helper: load model, format metadata report. Mirrors the
// `inspectQnnBinaryReport` shape so the LocalAiApp probe stays symmetric.
std::string mnnInspectReport(const std::string& mnnPath);

// Build the CLIP text-encoder input embedding by indexing fp16 token-embedding
// table at `tokenEmbBinPath` (shape [vocab, dim]) and summing in the fp32
// position-embedding table at `posEmbBinPath` (shape [seqLen, dim]). Returns
// fp32 vector of length tokens.size() * dim. Throws std::runtime_error on I/O
// failure. xororz's bundle uses dim=768, seqLen=77, vocab=49408.
std::vector<float> clipEmbedTokens(const std::vector<int32_t>& tokens,
                                   const std::string&          tokenEmbBinPath,
                                   const std::string&          posEmbBinPath,
                                   int                         embedDim = 768);

}  // namespace imagegen
