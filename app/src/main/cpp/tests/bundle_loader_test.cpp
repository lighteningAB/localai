// Host-side test for imagegen::loadBundle. Requires an extracted xororz bundle
// at the path given by the env var IMAGEGEN_TEST_BUNDLE_DIR. If unset, the test
// is reported as SKIPPED (exit 0) — CI / fresh dev machines don't need to keep
// the ~1 GB bundle around.
//
// run_host_tests.sh extracts a small subset of the bundle on demand; see that
// script for the setup details.

#include "../bundle_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int gFailures = 0;
}

#define EXPECT(cond, label)                                                       \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "FAIL  %s\n", (label));                          \
            ++gFailures;                                                          \
        } else {                                                                  \
            std::fprintf(stdout, "OK    %s\n", (label));                          \
        }                                                                         \
    } while (0)

int main() {
    const char* envDir = std::getenv("IMAGEGEN_TEST_BUNDLE_DIR");
    if (!envDir || !*envDir) {
        std::fprintf(stdout, "SKIP  loadBundle: IMAGEGEN_TEST_BUNDLE_DIR not set\n");
        return 0;
    }
    std::string dir = envDir;
    imagegen::Bundle b;
    std::string err;
    bool ok = imagegen::loadBundle(dir, b, err);
    if (!ok) {
        std::fprintf(stderr, "FAIL  loadBundle('%s') err=%s\n", dir.c_str(), err.c_str());
        return 1;
    }
    EXPECT(b.root == dir,                  "bundle.root matches input");
    EXPECT(!b.tokenizerJson.empty(),       "tokenizer.json path populated");
    EXPECT(!b.clipMnn.empty(),             "clip_v2.mnn path populated");
    EXPECT(!b.posEmbBin.empty(),           "pos_emb.bin path populated");
    EXPECT(!b.tokenEmbBin.empty(),         "token_emb.bin path populated");
    EXPECT(!b.vaeEncoderBin.empty(),       "vae_encoder.bin path populated");
    EXPECT(!b.vaeDecoderBin.empty(),       "vae_decoder.bin path populated");
    EXPECT(!b.unetBin.empty(),             "unet.bin path populated");
    EXPECT(!b.patchFiles.empty(),          "at least one .patch file found");

    // Negative test: missing dir should fail clearly.
    imagegen::Bundle b2;
    std::string err2;
    bool nok = imagegen::loadBundle(dir + "/__definitely_not_here__", b2, err2);
    EXPECT(!nok && !err2.empty(),          "loadBundle on missing dir returns false with error");

    std::fprintf(stdout, "\n%s — %d failure(s)\n",
                 (gFailures == 0 ? "PASS" : "FAIL"), gFailures);
    return gFailures == 0 ? 0 : 1;
}
