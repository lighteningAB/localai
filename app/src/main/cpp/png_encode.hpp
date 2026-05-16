#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imagegen {

// Implemented in png_encode.cpp. RGB buffer is packed HWC, 3 bytes per pixel,
// row-major. `pngOut` is overwritten with a complete PNG byte stream on
// success. Returns false with a reason in `error` if encoding fails (the only
// realistic cause is OOM in stbi_write_png_to_mem).
bool encodeRgbToPng(int width, int height,
                    const std::vector<uint8_t>& rgb,
                    std::vector<uint8_t>& pngOut,
                    std::string& error);

}  // namespace imagegen
