#include "mnn_session.hpp"

#include <android/log.h>

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

namespace imagegen {

namespace {

constexpr const char* kTag = "mnn_session";

void logI(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, ap);
    va_end(ap);
}

}  // namespace

struct MnnSession::Impl {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session*                     session = nullptr;
    MNNForwardType                    forward = MNN_FORWARD_CPU;
    std::vector<MnnTensorInfo>        inputs;
    std::vector<MnnTensorInfo>        outputs;

    ~Impl() {
        if (net && session) net->releaseSession(session);
    }
};

MnnSession::MnnSession()  : impl_(std::make_unique<Impl>()) {}
MnnSession::~MnnSession() = default;

namespace {

MnnTensorInfo tensorInfo(const std::string& name, const MNN::Tensor* t) {
    MnnTensorInfo info;
    info.name        = name;
    info.elementBits = t->getType().bits;
    info.isFloat     = (t->getType().code == halide_type_float);
    for (int d : t->shape()) info.dims.push_back(d);
    return info;
}

}  // namespace

bool MnnSession::load(const std::string& path, bool preferOpenCL, std::string& error) {
    impl_->net.reset(MNN::Interpreter::createFromFile(path.c_str()));
    if (!impl_->net) {
        error = "MNN::Interpreter::createFromFile failed: " + path;
        return false;
    }

    MNN::ScheduleConfig cfg{};
    cfg.type      = preferOpenCL ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
    cfg.numThread = 4;
    impl_->session = impl_->net->createSession(cfg);
    impl_->forward = static_cast<MNNForwardType>(cfg.type);
    if (!impl_->session && preferOpenCL) {
        logI("OpenCL session failed, falling back to CPU");
        cfg.type       = MNN_FORWARD_CPU;
        impl_->session = impl_->net->createSession(cfg);
        impl_->forward = MNN_FORWARD_CPU;
    }
    if (!impl_->session) {
        error = "MNN createSession failed (both OpenCL and CPU)";
        return false;
    }

    for (const auto& kv : impl_->net->getSessionInputAll(impl_->session)) {
        impl_->inputs.push_back(tensorInfo(kv.first, kv.second));
    }
    for (const auto& kv : impl_->net->getSessionOutputAll(impl_->session)) {
        impl_->outputs.push_back(tensorInfo(kv.first, kv.second));
    }
    return true;
}

namespace {

// IEEE-754 half-precision → single-precision. Standalone since we use it both
// for MNN output unpacking and for token-embedding-table lookup.
float halfToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            ++exp;
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(float));
    return r;
}

}  // namespace

bool MnnSession::runForward(const std::vector<float>& input,
                            std::vector<float>&       output,
                            std::string&              error) {
    if (!impl_->session) { error = "session not initialized"; return false; }

    MNN::Tensor* in = impl_->net->getSessionInput(impl_->session, nullptr);
    if (!in) { error = "no input tensor"; return false; }

    int expected = 1;
    for (int d : in->shape()) expected *= d;
    if (static_cast<int>(input.size()) != expected) {
        error = "input size mismatch: got " + std::to_string(input.size()) +
                " expected " + std::to_string(expected);
        return false;
    }
    if (in->getType().code != halide_type_float || in->getType().bits != 32) {
        error = "model input is not fp32 (code=" +
                std::to_string(in->getType().code) +
                " bits=" + std::to_string(in->getType().bits) + ")";
        return false;
    }
    auto hostIn = std::unique_ptr<MNN::Tensor>(
        MNN::Tensor::createHostTensorFromDevice(in, false));
    std::memcpy(hostIn->host<float>(), input.data(), input.size() * sizeof(float));
    in->copyFromHostTensor(hostIn.get());

    impl_->net->runSession(impl_->session);

    MNN::Tensor* out = impl_->net->getSessionOutput(impl_->session, nullptr);
    if (!out) { error = "no output tensor"; return false; }
    auto hostOut = std::unique_ptr<MNN::Tensor>(
        MNN::Tensor::createHostTensorFromDevice(out, false));
    out->copyToHostTensor(hostOut.get());

    int total = 1;
    for (int d : hostOut->shape()) total *= d;
    output.resize(static_cast<std::size_t>(total));

    if (hostOut->getType().code == halide_type_float && hostOut->getType().bits == 32) {
        std::memcpy(output.data(), hostOut->host<float>(), total * sizeof(float));
    } else if (hostOut->getType().code == halide_type_float && hostOut->getType().bits == 16) {
        const uint16_t* src = hostOut->host<uint16_t>();
        for (int i = 0; i < total; ++i) output[i] = halfToFloat(src[i]);
    } else {
        error = "output tensor dtype unsupported (code=" +
                std::to_string(hostOut->getType().code) +
                " bits=" + std::to_string(hostOut->getType().bits) + ")";
        return false;
    }
    return true;
}

std::vector<float> clipEmbedTokens(const std::vector<int32_t>& tokens,
                                   const std::string&          tokenEmbPath,
                                   const std::string&          posEmbPath,
                                   int                         embedDim) {
    // token_emb.bin: fp16 [vocab, embedDim]
    std::ifstream tf(tokenEmbPath, std::ios::binary | std::ios::ate);
    if (!tf) throw std::runtime_error("cannot open " + tokenEmbPath);
    std::streamsize tSize = tf.tellg();
    tf.seekg(0, std::ios::beg);
    if (tSize <= 0 || tSize % (embedDim * 2) != 0) {
        throw std::runtime_error("token_emb.bin size suspicious: " + std::to_string(tSize));
    }
    std::size_t vocab = static_cast<std::size_t>(tSize) / (static_cast<std::size_t>(embedDim) * 2);
    std::vector<uint16_t> tokTable(static_cast<std::size_t>(tSize) / 2);
    if (!tf.read(reinterpret_cast<char*>(tokTable.data()), tSize)) {
        throw std::runtime_error("read failed: " + tokenEmbPath);
    }

    // pos_emb.bin: fp32 [seqLen, embedDim]
    std::ifstream pf(posEmbPath, std::ios::binary | std::ios::ate);
    if (!pf) throw std::runtime_error("cannot open " + posEmbPath);
    std::streamsize pSize = pf.tellg();
    pf.seekg(0, std::ios::beg);
    if (pSize <= 0 || pSize % (embedDim * 4) != 0) {
        throw std::runtime_error("pos_emb.bin size suspicious: " + std::to_string(pSize));
    }
    std::size_t maxPos = static_cast<std::size_t>(pSize) / (static_cast<std::size_t>(embedDim) * 4);
    std::vector<float> posTable(static_cast<std::size_t>(pSize) / 4);
    if (!pf.read(reinterpret_cast<char*>(posTable.data()), pSize)) {
        throw std::runtime_error("read failed: " + posEmbPath);
    }

    std::vector<float> out(tokens.size() * static_cast<std::size_t>(embedDim));
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const int tid = tokens[i];
        if (tid < 0 || static_cast<std::size_t>(tid) >= vocab) {
            throw std::runtime_error("token id " + std::to_string(tid) +
                                     " out of vocab range " + std::to_string(vocab));
        }
        const std::size_t pos = i < maxPos ? i : (maxPos - 1);
        const uint16_t* tokRow = tokTable.data() + static_cast<std::size_t>(tid) * embedDim;
        const float*    posRow = posTable.data() + pos * embedDim;
        float*          outRow = out.data() + i * embedDim;
        for (int j = 0; j < embedDim; ++j) {
            outRow[j] = halfToFloat(tokRow[j]) + posRow[j];
        }
    }
    return out;
}

const std::vector<MnnTensorInfo>& MnnSession::inputs() const  { return impl_->inputs; }
const std::vector<MnnTensorInfo>& MnnSession::outputs() const { return impl_->outputs; }

std::string MnnSession::backendName() const {
    switch (impl_->forward) {
        case MNN_FORWARD_CPU:    return "CPU";
        case MNN_FORWARD_OPENCL: return "OpenCL";
        case MNN_FORWARD_OPENGL: return "OpenGL";
        case MNN_FORWARD_VULKAN: return "Vulkan";
        default:                 return "unknown";
    }
}

namespace {

void appendTensor(std::ostringstream& os, const char* role, const MnnTensorInfo& t) {
    os << "    " << role << " '" << t.name << "' "
       << (t.isFloat ? "fp" : "int") << t.elementBits << "[";
    for (std::size_t i = 0; i < t.dims.size(); ++i) {
        if (i) os << 'x';
        os << t.dims[i];
    }
    os << "]\n";
}

}  // namespace

std::string mnnInspectReport(const std::string& mnnPath) {
    MnnSession s;
    std::string err;
    if (!s.load(mnnPath, /*preferOpenCL=*/true, err)) {
        return "ERROR: load: " + err;
    }
    std::ostringstream os;
    os << "MNN model: " << mnnPath << "\n";
    os << "backend: " << s.backendName() << "\n";
    os << "inputs:\n";
    for (const auto& t : s.inputs())  appendTensor(os, "in ", t);
    os << "outputs:\n";
    for (const auto& t : s.outputs()) appendTensor(os, "out", t);
    return os.str();
}

}  // namespace imagegen
