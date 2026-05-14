// Host-side test for imagegen::Tokenizer. Loads a real CLIP tokenizer.json
// from the env var IMAGEGEN_TEST_TOKENIZER_JSON (defaults to /tmp/imagegen-tokenizer.json)
// and verifies encoding against fixed Python tokenizers reference outputs.

#include "../tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int gFailures = 0;

void expectIdsPrefix(const char* label,
                     const std::vector<int32_t>& got,
                     const std::vector<int32_t>& expectedPrefix) {
    bool ok = got.size() >= expectedPrefix.size();
    if (ok) {
        for (std::size_t i = 0; i < expectedPrefix.size(); ++i) {
            if (got[i] != expectedPrefix[i]) { ok = false; break; }
        }
    }
    if (!ok) {
        std::fprintf(stderr, "FAIL  %s\n  got=[", label);
        for (std::size_t i = 0; i < std::min(expectedPrefix.size() + 2, got.size()); ++i) {
            std::fprintf(stderr, "%d%s", got[i], i + 1 < got.size() ? "," : "");
        }
        std::fprintf(stderr, "...]\n  exp=[");
        for (std::size_t i = 0; i < expectedPrefix.size(); ++i) {
            std::fprintf(stderr, "%d%s", expectedPrefix[i], i + 1 < expectedPrefix.size() ? "," : "");
        }
        std::fprintf(stderr, "]\n");
        ++gFailures;
    } else {
        std::fprintf(stdout, "OK    %s\n", label);
    }
}

void expectLen(const char* label, const std::vector<int32_t>& got, std::size_t want) {
    if (got.size() != want) {
        std::fprintf(stderr, "FAIL  %s: got len=%zu want=%zu\n", label, got.size(), want);
        ++gFailures;
    } else {
        std::fprintf(stdout, "OK    %s (len=%zu)\n", label, want);
    }
}

}  // namespace

int main() {
    const char* envJson = std::getenv("IMAGEGEN_TEST_TOKENIZER_JSON");
    std::string path = envJson ? envJson : "/tmp/imagegen-tokenizer.json";

    imagegen::Tokenizer tok(path);
    std::fprintf(stdout, "INFO  vocab=%zu merges=%zu bos=%d eos=%d\n",
                 tok.vocabSize(), tok.mergeCount(), tok.bosId(), tok.eosId());

    // Reference outputs captured directly from the Python `tokenizers` library
    // (HuggingFace) against the same tokenizer.json. The tokenizer's natural
    // output has no padding; here we test the padded-to-77 form our `encode`
    // returns.
    constexpr int kPad = 77;

    {
        auto ids = tok.encode("a cat", kPad);
        expectLen("encode('a cat')", ids, 77);
        expectIdsPrefix("'a cat' → [BOS, 320, 2368, EOS]",
                        ids, {49406, 320, 2368, 49407});
        // Padding from position 4 onwards should all be EOS.
        bool pad_ok = true;
        for (std::size_t i = 4; i < ids.size(); ++i) {
            if (ids[i] != 49407) { pad_ok = false; break; }
        }
        if (!pad_ok) { std::fprintf(stderr, "FAIL  'a cat' EOS padding broken\n"); ++gFailures; }
        else         { std::fprintf(stdout, "OK    'a cat' EOS-padded to 77\n"); }
    }
    {
        auto ids = tok.encode("a photo of a cat", kPad);
        expectIdsPrefix("'a photo of a cat'",
                        ids, {49406, 320, 1125, 539, 320, 2368, 49407});
    }
    {
        auto ids = tok.encode("", kPad);
        expectIdsPrefix("'' → [BOS, EOS, EOS, ...]",
                        ids, {49406, 49407, 49407});
    }
    {
        auto ids = tok.encode("A Cat!", kPad);
        // Python ref: [49406, 320, 2368, 256, 49407]
        expectIdsPrefix("'A Cat!' (case + punctuation)",
                        ids, {49406, 320, 2368, 256, 49407});
    }
    {
        auto ids = tok.encode("the brown fox jumps over the lazy dog", kPad);
        expectIdsPrefix("'the brown fox jumps over the lazy dog'",
                        ids, {49406, 518, 2866, 3240, 18911, 962, 518, 10753, 1929, 49407});
    }

    std::fprintf(stdout, "\n%s — %d failure(s)\n",
                 (gFailures == 0 ? "PASS" : "FAIL"), gFailures);
    return gFailures == 0 ? 0 : 1;
}
