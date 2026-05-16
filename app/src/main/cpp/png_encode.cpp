// Single translation unit that owns stb_image_write's implementation, isolating
// the (warning-noisy) public-domain headers from the rest of the project. The
// stb-vendored copy lives under 3rdparty/stb/stb_image_write.h — replaced only
// by re-fetching upstream.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include "png_encode.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace imagegen {

namespace {
void appendToVector(void* context, void* data, int size) {
    auto* v = static_cast<std::vector<uint8_t>*>(context);
    const auto* p = static_cast<const uint8_t*>(data);
    v->insert(v->end(), p, p + size);
}
}  // namespace

// Encode an HWC interleaved RGB uint8 buffer to PNG bytes.
// Returns true on success. `rgb.size()` must equal width*height*3.
bool encodeRgbToPng(int width, int height,
                    const std::vector<uint8_t>& rgb,
                    std::vector<uint8_t>& pngOut,
                    std::string& error) {
    const std::size_t expected = static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height) * 3u;
    if (rgb.size() != expected) {
        error = "encodeRgbToPng: rgb size " + std::to_string(rgb.size()) +
                " != width*height*3 (" + std::to_string(expected) + ")";
        return false;
    }
    pngOut.clear();
    pngOut.reserve(expected / 2);
    const int rc = stbi_write_png_to_func(appendToVector, &pngOut,
                                          width, height, /*comp=*/3,
                                          rgb.data(),
                                          /*stride_bytes=*/width * 3);
    if (rc == 0) {
        error = "stbi_write_png_to_func failed";
        return false;
    }
    return true;
}

}  // namespace imagegen
