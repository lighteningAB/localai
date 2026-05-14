#include "tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>

// nlohmann/json triggers a handful of warnings on -Wpedantic builds; silence
// them around the include only.
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wpedantic"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include "3rdparty/nlohmann/json.hpp"
#if defined(__clang__)
#  pragma clang diagnostic pop
#endif

namespace imagegen {

namespace {

int utf8Len(unsigned char c) {
    if      ((c & 0x80) == 0x00) return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

std::string codepointToUtf8(int cp) {
    std::string r;
    if (cp < 0x80) {
        r += static_cast<char>(cp);
    } else if (cp < 0x800) {
        r += static_cast<char>(0xC0 | (cp >> 6));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        r += static_cast<char>(0xE0 | (cp >> 12));
        r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        r += static_cast<char>(0xF0 | (cp >> 18));
        r += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return r;
}

}  // namespace

std::array<std::string, 256> Tokenizer::buildByteEncoder() {
    // GPT-2 / RoBERTa bytes_to_unicode mapping. Maps each byte to a printable
    // Unicode codepoint so BPE never sees control bytes or whitespace.
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b)  bs.push_back(b);  // 33..126
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::array<std::string, 256> out;
    for (std::size_t i = 0; i < bs.size(); ++i) {
        out[static_cast<std::size_t>(bs[i])] = codepointToUtf8(cs[i]);
    }
    return out;
}

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Tokenizer: cannot open " + path);
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("Tokenizer: JSON parse error in " + path + ": " + e.what());
    }

    auto& model = j.at("model");
    if (!model.contains("vocab") || !model.contains("merges")) {
        throw std::runtime_error("Tokenizer: model.vocab or model.merges missing in " + path);
    }
    const auto& vocabJ = model["vocab"];
    vocab_.reserve(vocabJ.size());
    for (auto it = vocabJ.begin(); it != vocabJ.end(); ++it) {
        vocab_.emplace(it.key(), it.value().get<int32_t>());
    }

    const auto& mergesJ = model["merges"];
    mergeRank_.reserve(mergesJ.size());
    int32_t rank = 0;
    for (const auto& m : mergesJ) {
        // tokenizer.json merges are arrays of [left, right] strings.
        if (!m.is_array() || m.size() != 2) continue;
        std::string left  = m[0].get<std::string>();
        std::string right = m[1].get<std::string>();
        mergeRank_[left + "\x1F" + right] = rank++;
    }

    endOfWordSuffix_ = model.value("end_of_word_suffix", "</w>");

    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& at : j["added_tokens"]) {
            const std::string content = at.value("content", "");
            const int32_t id          = at.value("id", -1);
            if (id < 0) continue;
            if (content == "<|startoftext|>") bosId_ = id;
            if (content == "<|endoftext|>")   eosId_ = id;
        }
    }

    byteEncoder_ = buildByteEncoder();
}

std::vector<std::string> Tokenizer::preTokenize(const std::string& s) const {
    // Implements the CLIP Split regex:
    //   'res|'ves|'lls|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
    // with ASCII \p{L}/\p{N} classification + multibyte UTF-8 treated as letters.
    static const char* kContractions[] = {
        "'re", "'ve", "'ll", "'s", "'t", "'m", "'d"
    };

    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);

        // 1. Whitespace — dropped (Split with invert=True keeps only matches).
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++i; continue; }

        // 2. Contraction suffix.
        if (c == '\'') {
            bool matched = false;
            for (const char* ct : kContractions) {
                std::size_t L = std::strlen(ct);
                if (i + L <= s.size() && std::memcmp(s.data() + i, ct, L) == 0) {
                    out.emplace_back(ct, L);
                    i += L;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
            // Fall through: treat apostrophe as punctuation.
        }

        // 3. Letter run [\p{L}]+
        auto isLetterByte = [](unsigned char cc) {
            return (cc >= 'a' && cc <= 'z') || cc >= 0x80;
        };
        if (isLetterByte(c)) {
            std::size_t start = i;
            while (i < s.size()) {
                unsigned char cc = static_cast<unsigned char>(s[i]);
                if (!isLetterByte(cc)) break;
                if (cc >= 0x80) {
                    int len = utf8Len(cc);
                    if (i + static_cast<std::size_t>(len) > s.size()) len = 1;
                    i += static_cast<std::size_t>(len);
                } else {
                    ++i;
                }
            }
            out.emplace_back(s, start, i - start);
            continue;
        }

        // 4. Single digit [\p{N}]
        if (c >= '0' && c <= '9') {
            out.emplace_back(s, i, 1);
            ++i;
            continue;
        }

        // 5. Punctuation run [^\s\p{L}\p{N}]+
        std::size_t start = i;
        while (i < s.size()) {
            unsigned char cc = static_cast<unsigned char>(s[i]);
            if (cc == ' ' || cc == '\t' || cc == '\n' || cc == '\r') break;
            if (isLetterByte(cc))                                    break;
            if (cc >= '0' && cc <= '9')                              break;
            ++i;
        }
        out.emplace_back(s, start, i - start);
    }
    return out;
}

std::string Tokenizer::byteEncode(const std::string& raw) const {
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) {
        out += byteEncoder_[c];
    }
    return out;
}

std::vector<std::string> Tokenizer::bpe(const std::string& encoded) const {
    // Split into individual Unicode characters (each a UTF-8 string).
    std::vector<std::string> symbols;
    symbols.reserve(encoded.size());
    std::size_t i = 0;
    while (i < encoded.size()) {
        int len = utf8Len(static_cast<unsigned char>(encoded[i]));
        if (i + static_cast<std::size_t>(len) > encoded.size()) len = 1;
        symbols.emplace_back(encoded, i, static_cast<std::size_t>(len));
        i += static_cast<std::size_t>(len);
    }
    if (symbols.empty()) return {};
    // CLIP word-end marker on the final symbol.
    symbols.back() += endOfWordSuffix_;

    while (symbols.size() >= 2) {
        int bestRank = INT_MAX;
        std::size_t bestI = 0;
        for (std::size_t j = 0; j + 1 < symbols.size(); ++j) {
            std::string key = symbols[j] + "\x1F" + symbols[j + 1];
            auto it = mergeRank_.find(key);
            if (it != mergeRank_.end() && it->second < bestRank) {
                bestRank = it->second;
                bestI = j;
            }
        }
        if (bestRank == INT_MAX) break;
        symbols[bestI] = symbols[bestI] + symbols[bestI + 1];
        symbols.erase(symbols.begin() + static_cast<long>(bestI) + 1);
    }
    return symbols;
}

std::vector<int32_t> Tokenizer::encode(const std::string& prompt, int padLen) const {
    if (padLen < 2) {
        throw std::invalid_argument("Tokenizer::encode: padLen must be >= 2 (BOS+EOS)");
    }

    // ASCII-only normalization: collapse whitespace + lowercase.
    std::string norm;
    norm.reserve(prompt.size());
    bool prevSpace = false;
    for (char raw : prompt) {
        unsigned char c = static_cast<unsigned char>(raw);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!prevSpace && !norm.empty()) norm += ' ';
            prevSpace = true;
        } else {
            if (c >= 'A' && c <= 'Z') c += 32;
            norm += static_cast<char>(c);
            prevSpace = false;
        }
    }
    while (!norm.empty() && norm.back() == ' ') norm.pop_back();

    std::vector<std::string> chunks = preTokenize(norm);

    std::vector<int32_t> ids;
    ids.reserve(static_cast<std::size_t>(padLen));
    ids.push_back(bosId_);
    for (const auto& c : chunks) {
        std::string enc = byteEncode(c);
        auto pieces = bpe(enc);
        for (const auto& p : pieces) {
            auto it = vocab_.find(p);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            }
            // Else: unknown token (e.g. byte-level fallback miss). Skip — CLIP
            // BPE with end_of_word_suffix is closed over byteEncoder output for
            // valid UTF-8 input.
        }
    }
    ids.push_back(eosId_);

    if (static_cast<int>(ids.size()) > padLen) {
        ids.resize(static_cast<std::size_t>(padLen));
        ids[static_cast<std::size_t>(padLen - 1)] = eosId_;
    } else {
        ids.resize(static_cast<std::size_t>(padLen), eosId_);
    }
    return ids;
}

}  // namespace imagegen
