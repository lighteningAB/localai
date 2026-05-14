#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace imagegen {

// CLIP byte-level BPE tokenizer compatible with the tokenizer.json shipped in
// xororz SD-QNN bundles (SD 1.5 / CLIP ViT-L/14, vocab=49408).
//
// Hand-rolled (no Rust dependency). Encoding flow matches HF tokenizers'
// pipeline for this tokenizer.json:
//   1. Normalize: NFC → collapse whitespace → lowercase
//      ASCII fast-path. NFC + Unicode lowercase are NOT implemented; non-ASCII
//      input may tokenize slightly differently from the Python ref. Sufficient
//      for English SD prompts (the common case).
//   2. Pre-tokenize via the CLIP regex (\p{L}/\p{N} are approximated by ASCII +
//      "any multibyte UTF-8 is a letter"). Whitespace is dropped.
//   3. ByteLevel: bytes_to_unicode encode each chunk (GPT-2 / RoBERTa style).
//   4. BPE merges (greedy by rank), with `</w>` appended to the last symbol.
//   5. Vocab lookup → int IDs.
//   6. Post-process: prepend BOS (<|startoftext|>=49406), append EOS
//      (<|endoftext|>=49407), pad/truncate to `padLen` (default 77) with EOS.
class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizerJsonPath);

    // Encode `prompt` to a fixed-length token id sequence (default 77 for SD 1.5).
    // If the natural encoding exceeds `padLen`, it is truncated and the last
    // slot is forced to EOS. If shorter, the remainder is padded with EOS.
    std::vector<int32_t> encode(const std::string& prompt, int padLen = 77) const;

    int32_t bosId() const { return bosId_; }
    int32_t eosId() const { return eosId_; }
    std::size_t vocabSize() const { return vocab_.size(); }
    std::size_t mergeCount() const { return mergeRank_.size(); }

private:
    std::unordered_map<std::string, int32_t> vocab_;
    // Keyed by "left\x1Fright" so the separator can't collide with token text.
    std::unordered_map<std::string, int32_t> mergeRank_;
    std::array<std::string, 256> byteEncoder_;
    std::string endOfWordSuffix_;
    int32_t bosId_ = 49406;
    int32_t eosId_ = 49407;

    static std::array<std::string, 256> buildByteEncoder();
    std::vector<std::string> preTokenize(const std::string& s) const;
    std::string byteEncode(const std::string& raw) const;
    std::vector<std::string> bpe(const std::string& encoded) const;
};

}  // namespace imagegen
