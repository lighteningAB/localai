#include "bundle_loader.hpp"

#include <sys/stat.h>

#include <cstring>
#include <dirent.h>
#include <sstream>

namespace imagegen {

namespace {

bool fileExists(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    return S_ISREG(st.st_mode);
}

bool dirExists(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

std::string join(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

bool requireFile(const std::string& dir, const char* name, std::string& out, std::string& error) {
    out = join(dir, name);
    if (!fileExists(out)) {
        std::ostringstream msg;
        msg << "missing required file: " << out;
        error = msg.str();
        return false;
    }
    return true;
}

bool endsWith(const std::string& s, const char* suffix) {
    size_t slen = s.size();
    size_t xlen = std::strlen(suffix);
    return slen >= xlen && std::memcmp(s.data() + (slen - xlen), suffix, xlen) == 0;
}

}  // namespace

bool loadBundle(const std::string& rootDir, Bundle& out, std::string& error) {
    if (!dirExists(rootDir)) {
        std::ostringstream msg;
        msg << "bundle root is not a directory: " << rootDir;
        error = msg.str();
        return false;
    }

    out.root = rootDir;

    if (!requireFile(rootDir, "tokenizer.json",  out.tokenizerJson,  error)) return false;
    if (!requireFile(rootDir, "clip_v2.mnn",     out.clipMnn,        error)) return false;
    if (!requireFile(rootDir, "pos_emb.bin",     out.posEmbBin,      error)) return false;
    if (!requireFile(rootDir, "token_emb.bin",   out.tokenEmbBin,    error)) return false;
    if (!requireFile(rootDir, "vae_encoder.bin", out.vaeEncoderBin,  error)) return false;
    if (!requireFile(rootDir, "vae_decoder.bin", out.vaeDecoderBin,  error)) return false;
    if (!requireFile(rootDir, "unet.bin",        out.unetBin,        error)) return false;

    // Optional: enumerate .patch files (resolution-specific UNet overlays).
    if (DIR* d = ::opendir(rootDir.c_str())) {
        struct dirent* ent;
        while ((ent = ::readdir(d)) != nullptr) {
            std::string name = ent->d_name;
            if (endsWith(name, ".patch")) {
                out.patchFiles.push_back(join(rootDir, name));
            }
        }
        ::closedir(d);
    }

    return true;
}

}  // namespace imagegen
