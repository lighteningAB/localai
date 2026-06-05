#include "qnn_session.hpp"

#include <android/log.h>
#include <dlfcn.h>

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <sstream>

#if defined(IMAGEGEN_HAS_QNN)
#include <QnnBackend.h>
#include <QnnContext.h>
#include <QnnDevice.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnTypes.h>
#include <HTP/QnnHtpDevice.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>
#endif

namespace imagegen {

// -----------------------------------------------------------------------------
// Public dtype helpers — defined here (and not in the stub branch below) so
// they remain available even when QAIRT isn't configured. The values mirror
// QNN_DATATYPE_* numerically: see QnnTypes.h.
// -----------------------------------------------------------------------------

std::size_t bytesPerDataType(uint32_t dt) {
    // High nibble of the Qnn_DataType_t enum encodes category; low nibble
    // encodes bit-width / 8. We don't pull in QnnTypes.h on the stub branch,
    // so use a switch over the documented numeric values.
    switch (dt) {
        case 0x0008: return 1;   // INT_8
        case 0x0016: return 2;   // INT_16
        case 0x0032: return 4;   // INT_32
        case 0x0064: return 8;   // INT_64
        case 0x0108: return 1;   // UINT_8
        case 0x0116: return 2;   // UINT_16
        case 0x0132: return 4;   // UINT_32
        case 0x0164: return 8;   // UINT_64
        case 0x0216: return 2;   // FLOAT_16
        case 0x0232: return 4;   // FLOAT_32
        case 0x0264: return 8;   // FLOAT_64
        case 0x0308: return 1;   // SFIXED_POINT_8
        case 0x0316: return 2;   // SFIXED_POINT_16
        case 0x0332: return 4;   // SFIXED_POINT_32
        case 0x0408: return 1;   // UFIXED_POINT_8
        case 0x0416: return 2;   // UFIXED_POINT_16
        case 0x0432: return 4;   // UFIXED_POINT_32
        case 0x0508: return 1;   // BOOL_8
        default:     return 0;   // unsupported / sub-byte
    }
}

std::size_t tensorByteSize(const QnnTensorInfo& info) {
    std::size_t bytes = bytesPerDataType(info.dataType);
    if (bytes == 0) return 0;
    std::size_t elements = 1;
    for (uint32_t d : info.dims) elements *= d;
    return elements * bytes;
}

namespace {

constexpr const char* kTag = "qnn_session";

void logI(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, ap);
    va_end(ap);
}

bool readEntireFile(const std::string& path, std::vector<uint8_t>& out, std::string& error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        error = "cannot open: " + path;
        return false;
    }
    std::streamsize size = f.tellg();
    if (size <= 0) {
        error = "empty or unreadable file: " + path;
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    if (!f.read(reinterpret_cast<char*>(out.data()), size)) {
        error = "read failed: " + path;
        return false;
    }
    return true;
}

}  // namespace

#if !defined(IMAGEGEN_HAS_QNN)

// -----------------------------------------------------------------------------
// Stub build (no QAIRT). Methods report the toolchain config.
// -----------------------------------------------------------------------------

struct QnnSession::Impl {};

QnnSession::QnnSession() = default;
QnnSession::~QnnSession() = default;

bool QnnSession::initialize(std::string& error) {
    error = "QnnSession compiled without QAIRT (QNN_SDK_ROOT was unset at build time). "
            "Set qnn.sdk.root in local.properties and rebuild.";
    return false;
}

bool QnnSession::inspectBinary(const std::string& /*path*/, std::string& error) {
    error = "QnnSession not built with QAIRT.";
    return false;
}

bool QnnSession::instantiate(std::string& error) {
    error = "QnnSession not built with QAIRT.";
    return false;
}

bool QnnSession::execute(std::size_t /*idx*/,
                         const std::vector<std::vector<uint8_t>>& /*inputs*/,
                         std::vector<std::vector<uint8_t>>&       /*outputs*/,
                         std::string&                             error) {
    error = "QnnSession not built with QAIRT.";
    return false;
}

bool QnnSession::initialized()  const { return false; }
bool QnnSession::instantiated() const { return false; }

namespace {
const std::vector<QnnGraphInfo>& emptyGraphs() {
    static const std::vector<QnnGraphInfo> g;
    return g;
}
}  // namespace

const std::vector<QnnGraphInfo>& QnnSession::graphs() const { return emptyGraphs(); }
std::string QnnSession::interfaceVersion() const { return {}; }

std::string inspectQnnBinaryReport(const std::string& path) {
    return std::string("ERROR: QnnSession not built with QAIRT (path=") + path + ")";
}

std::string probeQnnBinaryLoadReport(const std::string& path) {
    return std::string("ERROR: QnnSession not built with QAIRT (path=") + path + ")";
}

#else  // IMAGEGEN_HAS_QNN

// -----------------------------------------------------------------------------
// Real implementation against the QAIRT 2.x QnnInterface / QnnSystemInterface.
// -----------------------------------------------------------------------------

struct QnnSession::Impl {
    void*                              libHtp     = nullptr;
    void*                              libSystem  = nullptr;
    const QnnInterface_t*              ifaceProvider = nullptr;
    const QnnSystemInterface_t*        sysProvider   = nullptr;
    QnnSystemContext_Handle_t          sysCtx     = nullptr;

    std::vector<uint8_t>               binary;
    std::vector<QnnGraphInfo>          graphs;            // our deep-copied metadata
    std::string                        interfaceVersion;
    bool                               initialized  = false;
    bool                               instantiated = false;

    // Live runtime handles (populated by instantiate()).
    Qnn_LogHandle_t                    logHandle     = nullptr;
    Qnn_BackendHandle_t                backendHandle = nullptr;
    Qnn_DeviceHandle_t                 deviceHandle  = nullptr;
    Qnn_ContextHandle_t                contextHandle = nullptr;
    std::vector<Qnn_GraphHandle_t>     graphHandles;       // 1:1 with graphs[]

    // Pointers into the binaryInfo blob — used in execute() as struct templates
    // (we need quantizeParams + the version-correct layout that the metadata
    // reader produced). Lifetime is tied to sysCtx + binary; both kept alive
    // for the session.
    std::vector<Qnn_Tensor_t*>         graphInputTemplates;   // numGraphs entries
    std::vector<Qnn_Tensor_t*>         graphOutputTemplates;  // numGraphs entries
    std::vector<uint32_t>              graphNumInputs;
    std::vector<uint32_t>              graphNumOutputs;

    ~Impl() {
        if (contextHandle && ifaceProvider) {
            iface().contextFree(contextHandle, /*profile=*/nullptr);
            contextHandle = nullptr;
        }
        if (deviceHandle && ifaceProvider) {
            iface().deviceFree(deviceHandle);
            deviceHandle = nullptr;
        }
        if (backendHandle && ifaceProvider) {
            iface().backendFree(backendHandle);
            backendHandle = nullptr;
        }
        if (logHandle && ifaceProvider) {
            iface().logFree(logHandle);
            logHandle = nullptr;
        }
        if (sysCtx && sysProvider) {
            sysProvider->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtx);
            sysCtx = nullptr;
        }
        if (libSystem) { ::dlclose(libSystem); libSystem = nullptr; }
        if (libHtp)    { ::dlclose(libHtp);    libHtp    = nullptr; }
    }

    const QNN_INTERFACE_VER_TYPE& iface() const {
        return ifaceProvider->QNN_INTERFACE_VER_NAME;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE& sysIface() const {
        return sysProvider->QNN_SYSTEM_INTERFACE_VER_NAME;
    }
};

QnnSession::QnnSession() : impl_(std::make_unique<Impl>()) {}
QnnSession::~QnnSession() = default;

bool QnnSession::initialized() const {
    return impl_ && impl_->initialized;
}

bool QnnSession::instantiated() const {
    return impl_ && impl_->instantiated;
}

const std::vector<QnnGraphInfo>& QnnSession::graphs() const {
    return impl_->graphs;
}

std::string QnnSession::interfaceVersion() const {
    return impl_ ? impl_->interfaceVersion : std::string{};
}

namespace {

using QnnInterface_GetProvidersFn       = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
using QnnSystemInterface_GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

// Pull the version-common Qnn_Tensor_t fields into a QnnTensorInfo.
QnnTensorInfo tensorInfoFrom(const Qnn_Tensor_t& t) {
    QnnTensorInfo info;
    const char*                 name  = nullptr;
    Qnn_DataType_t              dtype = QNN_DATATYPE_UNDEFINED;
    uint32_t                    rank  = 0;
    const uint32_t*             dims  = nullptr;
    const Qnn_QuantizeParams_t* qp    = nullptr;

    if (t.version == QNN_TENSOR_VERSION_1) {
        name  = t.v1.name;
        dtype = t.v1.dataType;
        rank  = t.v1.rank;
        dims  = t.v1.dimensions;
        qp    = &t.v1.quantizeParams;
    } else if (t.version == QNN_TENSOR_VERSION_2) {
        name  = t.v2.name;
        dtype = t.v2.dataType;
        rank  = t.v2.rank;
        dims  = t.v2.dimensions;
        qp    = &t.v2.quantizeParams;
    }

    if (name) info.name = name;
    info.dataType = static_cast<uint32_t>(dtype);
    info.dims.reserve(rank);
    for (uint32_t i = 0; i < rank; ++i) info.dims.push_back(dims ? dims[i] : 0);
    if (qp && qp->quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        info.quantScale  = qp->scaleOffsetEncoding.scale;
        info.quantOffset = qp->scaleOffsetEncoding.offset;
    }
    return info;
}

// Pull the version-common fields from a QnnSystemContext_GraphInfo_t into a QnnGraphInfo.
QnnGraphInfo graphInfoFrom(const QnnSystemContext_GraphInfo_t& gi) {
    QnnGraphInfo out;
    const char*    name           = nullptr;
    uint32_t       numInputs      = 0;
    Qnn_Tensor_t*  inputs         = nullptr;
    uint32_t       numOutputs     = 0;
    Qnn_Tensor_t*  outputs        = nullptr;

    if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        name       = gi.graphInfoV1.graphName;
        numInputs  = gi.graphInfoV1.numGraphInputs;
        inputs     = gi.graphInfoV1.graphInputs;
        numOutputs = gi.graphInfoV1.numGraphOutputs;
        outputs    = gi.graphInfoV1.graphOutputs;
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        name       = gi.graphInfoV2.graphName;
        numInputs  = gi.graphInfoV2.numGraphInputs;
        inputs     = gi.graphInfoV2.graphInputs;
        numOutputs = gi.graphInfoV2.numGraphOutputs;
        outputs    = gi.graphInfoV2.graphOutputs;
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        name       = gi.graphInfoV3.graphName;
        numInputs  = gi.graphInfoV3.numGraphInputs;
        inputs     = gi.graphInfoV3.graphInputs;
        numOutputs = gi.graphInfoV3.numGraphOutputs;
        outputs    = gi.graphInfoV3.graphOutputs;
    }

    if (name) out.name = name;
    out.inputs.reserve(numInputs);
    for (uint32_t i = 0; i < numInputs; ++i) {
        out.inputs.push_back(tensorInfoFrom(inputs[i]));
    }
    out.outputs.reserve(numOutputs);
    for (uint32_t i = 0; i < numOutputs; ++i) {
        out.outputs.push_back(tensorInfoFrom(outputs[i]));
    }
    return out;
}

// Extract numGraphs + graphs array out of a versioned BinaryInfo.
bool graphsFromBinaryInfo(const QnnSystemContext_BinaryInfo_t& bi,
                         uint32_t& numGraphs,
                         QnnSystemContext_GraphInfo_t*& graphs,
                         std::string& error) {
    if (bi.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        numGraphs = bi.contextBinaryInfoV1.numGraphs;
        graphs    = bi.contextBinaryInfoV1.graphs;
        return true;
    }
    if (bi.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        numGraphs = bi.contextBinaryInfoV2.numGraphs;
        graphs    = bi.contextBinaryInfoV2.graphs;
        return true;
    }
    if (bi.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        numGraphs = bi.contextBinaryInfoV3.numGraphs;
        graphs    = bi.contextBinaryInfoV3.graphs;
        return true;
    }
    error = "unknown QnnSystemContext_BinaryInfo version " + std::to_string(bi.version);
    return false;
}

}  // namespace

bool QnnSession::initialize(std::string& error) {
    if (impl_->initialized) return true;

    impl_->libHtp = ::dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!impl_->libHtp) {
        const char* dle = ::dlerror();
        error = std::string("dlopen libQnnHtp.so failed: ") + (dle ? dle : "<no error>");
        return false;
    }
    auto getInterfaceProviders = reinterpret_cast<QnnInterface_GetProvidersFn>(
        ::dlsym(impl_->libHtp, "QnnInterface_getProviders"));
    if (!getInterfaceProviders) {
        error = "dlsym QnnInterface_getProviders missing in libQnnHtp.so";
        return false;
    }
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    if (getInterfaceProviders(&providers, &numProviders) != QNN_SUCCESS ||
        !providers || numProviders == 0) {
        error = "QnnInterface_getProviders failed or returned no providers";
        return false;
    }
    for (uint32_t i = 0; i < numProviders; ++i) {
        const auto& v = providers[i]->apiVersion.coreApiVersion;
        if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
            impl_->ifaceProvider = providers[i];
            char buf[64];
            std::snprintf(buf, sizeof(buf), "core v%u.%u.%u", v.major, v.minor, v.patch);
            impl_->interfaceVersion = buf;
            break;
        }
    }
    if (!impl_->ifaceProvider) {
        error = "no compatible QnnInterface (need >= v" +
                std::to_string(QNN_API_VERSION_MAJOR) + "." +
                std::to_string(QNN_API_VERSION_MINOR) + ")";
        return false;
    }
    logI("QnnInterface_getProviders: %s", impl_->interfaceVersion.c_str());

    impl_->libSystem = ::dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!impl_->libSystem) {
        const char* dle = ::dlerror();
        error = std::string("dlopen libQnnSystem.so failed: ") + (dle ? dle : "<no error>");
        return false;
    }
    auto getSystemProviders = reinterpret_cast<QnnSystemInterface_GetProvidersFn>(
        ::dlsym(impl_->libSystem, "QnnSystemInterface_getProviders"));
    if (!getSystemProviders) {
        error = "dlsym QnnSystemInterface_getProviders missing in libQnnSystem.so";
        return false;
    }
    const QnnSystemInterface_t** sysProviders = nullptr;
    uint32_t numSysProviders = 0;
    if (getSystemProviders(&sysProviders, &numSysProviders) != QNN_SUCCESS ||
        !sysProviders || numSysProviders == 0) {
        error = "QnnSystemInterface_getProviders failed or returned no providers";
        return false;
    }
    for (uint32_t i = 0; i < numSysProviders; ++i) {
        const auto& v = sysProviders[i]->systemApiVersion;
        if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
            impl_->sysProvider = sysProviders[i];
            break;
        }
    }
    if (!impl_->sysProvider) {
        error = "no compatible QnnSystemInterface (need >= v" +
                std::to_string(QNN_SYSTEM_API_VERSION_MAJOR) + "." +
                std::to_string(QNN_SYSTEM_API_VERSION_MINOR) + ")";
        return false;
    }

    if (impl_->sysIface().systemContextCreate(&impl_->sysCtx) != QNN_SUCCESS || !impl_->sysCtx) {
        error = "QnnSystemContext_create failed";
        return false;
    }

    // Create a QNN log handle that forwards backend-internal messages to
    // logcat. Without this, deviceCreate/contextCreate failures only emit a
    // numeric error code with no context.
    static auto qnnLogTrampoline = +[](const char* fmt,
                                       QnnLog_Level_t level,
                                       uint64_t /*timestamp*/,
                                       va_list args) {
        int prio = ANDROID_LOG_INFO;
        switch (level) {
            case QNN_LOG_LEVEL_ERROR:   prio = ANDROID_LOG_ERROR; break;
            case QNN_LOG_LEVEL_WARN:    prio = ANDROID_LOG_WARN;  break;
            case QNN_LOG_LEVEL_INFO:    prio = ANDROID_LOG_INFO;  break;
            case QNN_LOG_LEVEL_VERBOSE: prio = ANDROID_LOG_VERBOSE; break;
            case QNN_LOG_LEVEL_DEBUG:   prio = ANDROID_LOG_DEBUG; break;
            default: break;
        }
        __android_log_vprint(prio, "qnn_runtime", fmt, args);
    };
    auto rcLog = impl_->iface().logCreate(qnnLogTrampoline,
                                          QNN_LOG_LEVEL_INFO,
                                          &impl_->logHandle);
    if (rcLog != QNN_SUCCESS || !impl_->logHandle) {
        logI("warning: QnnLog_create rc=0x%llx", (unsigned long long)rcLog);
        impl_->logHandle = nullptr;   // proceed without logging
    } else {
        logI("QNN runtime logging enabled at INFO");
    }

    impl_->initialized = true;
    return true;
}

bool QnnSession::inspectBinary(const std::string& path, std::string& error) {
    if (!impl_->initialized) {
        if (!initialize(error)) return false;
    }
    if (!readEntireFile(path, impl_->binary, error)) return false;
    logI("inspectBinary: %s (%zu bytes)", path.c_str(), impl_->binary.size());

    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    Qnn_ContextBinarySize_t              infoSize   = 0;
    auto rc = impl_->sysIface().systemContextGetBinaryInfo(
        impl_->sysCtx,
        impl_->binary.data(),
        impl_->binary.size(),
        &binaryInfo,
        &infoSize);
    if (rc != QNN_SUCCESS || !binaryInfo) {
        error = "QnnSystemContext_getBinaryInfo failed rc=" + std::to_string(rc);
        return false;
    }

    uint32_t                     numGraphs = 0;
    QnnSystemContext_GraphInfo_t* graphs   = nullptr;
    if (!graphsFromBinaryInfo(*binaryInfo, numGraphs, graphs, error)) return false;

    impl_->graphs.clear();
    impl_->graphs.reserve(numGraphs);
    impl_->graphInputTemplates.clear();
    impl_->graphOutputTemplates.clear();
    impl_->graphNumInputs.clear();
    impl_->graphNumOutputs.clear();
    impl_->graphInputTemplates.reserve(numGraphs);
    impl_->graphOutputTemplates.reserve(numGraphs);
    for (uint32_t i = 0; i < numGraphs; ++i) {
        impl_->graphs.push_back(graphInfoFrom(graphs[i]));

        // Cache the original Qnn_Tensor_t arrays — execute() will struct-copy
        // them and only flip clientBuf + memType + type before graphExecute.
        Qnn_Tensor_t* inputs   = nullptr;
        Qnn_Tensor_t* outputs  = nullptr;
        uint32_t      numIn    = 0;
        uint32_t      numOut   = 0;
        if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            inputs  = graphs[i].graphInfoV1.graphInputs;
            outputs = graphs[i].graphInfoV1.graphOutputs;
            numIn   = graphs[i].graphInfoV1.numGraphInputs;
            numOut  = graphs[i].graphInfoV1.numGraphOutputs;
        } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
            inputs  = graphs[i].graphInfoV2.graphInputs;
            outputs = graphs[i].graphInfoV2.graphOutputs;
            numIn   = graphs[i].graphInfoV2.numGraphInputs;
            numOut  = graphs[i].graphInfoV2.numGraphOutputs;
        } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            inputs  = graphs[i].graphInfoV3.graphInputs;
            outputs = graphs[i].graphInfoV3.graphOutputs;
            numIn   = graphs[i].graphInfoV3.numGraphInputs;
            numOut  = graphs[i].graphInfoV3.numGraphOutputs;
        }
        impl_->graphInputTemplates.push_back(inputs);
        impl_->graphOutputTemplates.push_back(outputs);
        impl_->graphNumInputs.push_back(numIn);
        impl_->graphNumOutputs.push_back(numOut);
    }
    logI("inspectBinary: %u graph(s)", numGraphs);
    return true;
}

bool QnnSession::instantiate(std::string& error) {
    if (!impl_->initialized) {
        error = "instantiate: call initialize() + inspectBinary() first";
        return false;
    }
    if (impl_->binary.empty()) {
        error = "instantiate: no binary loaded — call inspectBinary() first";
        return false;
    }
    if (impl_->instantiated) return true;

    // Create backend. Passing NULL log + config; defaults are fine for inference.
    {
        auto rc = impl_->iface().backendCreate(impl_->logHandle, /*config=*/nullptr,
                                               &impl_->backendHandle);
        if (rc != QNN_SUCCESS || !impl_->backendHandle) {
            error = "QnnBackend_create failed rc=0x" + std::to_string(rc);
            return false;
        }
    }
    logI("instantiate: backend created");

    // Probe platform info first — tells us what the runtime sees on the chip.
    {
        const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
        auto rcPlat = impl_->iface().deviceGetPlatformInfo(/*logger=*/nullptr,
                                                           &platformInfo);
        char b[32];
        std::snprintf(b, sizeof(b), "%llx", (unsigned long long)rcPlat);
        if (rcPlat == QNN_SUCCESS && platformInfo) {
            const auto& vp = platformInfo->v1;
            logI("instantiate: platformInfo numHwDevices=%u", vp.numHwDevices);
            for (uint32_t i = 0; i < vp.numHwDevices; ++i) {
                const auto& dev = vp.hwDevices[i].v1;
                logI("  device[%u] id=%u type=%u numCores=%u",
                     i, dev.deviceId, dev.deviceType, dev.numCores);
                auto* devExt = dev.deviceInfoExtension;
                if (devExt) {
                    const auto* htpDev = reinterpret_cast<const QnnHtpDevice_DeviceInfoExtension_t*>(devExt);
                    if (htpDev->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
                        logI("    onChip: socModel=%u arch=%d vtcm=%zu signedPd=%d",
                             htpDev->onChipDevice.socModel,
                             (int)htpDev->onChipDevice.arch,
                             htpDev->onChipDevice.vtcmSize,
                             (int)htpDev->onChipDevice.signedPdSupport);
                    }
                }
            }
            impl_->iface().deviceFreePlatformInfo(/*logger=*/nullptr, platformInfo);
        } else {
            logI("instantiate: deviceGetPlatformInfo rc=0x%s", b);
        }
    }

    // Create an HTP device. First attempt: NULL config (lets the runtime probe
    // the on-chip Hexagon revision). If the runtime rejects that (we've seen
    // QNN_DEVICE_ERROR_INVALID_CONFIG on this backend), retry with explicit
    // arch hints.
    auto tryDeviceCreate = [&](const QnnDevice_Config_t** cfg, const char* label) -> Qnn_ErrorHandle_t {
        impl_->deviceHandle = nullptr;
        auto rc = impl_->iface().deviceCreate(impl_->logHandle, cfg, &impl_->deviceHandle);
        char b[32];
        std::snprintf(b, sizeof(b), "%llx", (unsigned long long)rc);
        logI("instantiate: deviceCreate(%s) rc=0x%s handle=%p",
             label, b, impl_->deviceHandle);
        return rc;
    };

    // On SM8735 (Sun / 8s Gen 4), FastRPC defaults to **signed PD**, which a
    // debug-signed (untrusted) app is not allowed to use — the kernel logs
    // "Untrusted application trying to offload to signed PD". Explicitly
    // opt into UNSIGNED PD; this is what production SD inference apps do
    // when they're not OEM-signed. ARCH config is ignored by the real
    // target (runtime auto-detects V73), so we don't bother with it here.
    QnnHtpDevice_CustomConfig_t unsignedPdCfg{};
    unsignedPdCfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
    unsignedPdCfg.useSignedProcessDomain.deviceId             = 0;
    unsignedPdCfg.useSignedProcessDomain.useSignedProcessDomain = false;

    QnnDevice_Config_t unsignedPdDev{};
    unsignedPdDev.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    unsignedPdDev.customConfig = &unsignedPdCfg;

    const QnnDevice_Config_t* cfgUnsignedPd[] = { &unsignedPdDev, nullptr };
    auto rcDev = tryDeviceCreate(cfgUnsignedPd, "unsignedPd");

    if (rcDev != QNN_SUCCESS || !impl_->deviceHandle) {
        // Last-ditch: try NULL config (might pick a default that works).
        rcDev = tryDeviceCreate(nullptr, "null");
    }
    if (rcDev != QNN_SUCCESS || !impl_->deviceHandle) {
        char b[32];
        std::snprintf(b, sizeof(b), "%llx", (unsigned long long)rcDev);
        error = std::string("QnnDevice_create failed rc=0x") + b;
        impl_->iface().backendFree(impl_->backendHandle);
        impl_->backendHandle = nullptr;
        return false;
    }

    // Create context from the cached binary against our explicit V73 device.
    {
        auto rc = impl_->iface().contextCreateFromBinary(impl_->backendHandle,
                                                         impl_->deviceHandle,
                                                         /*config=*/nullptr,
                                                         impl_->binary.data(),
                                                         impl_->binary.size(),
                                                         &impl_->contextHandle,
                                                         /*profile=*/nullptr);
        if (rc != QNN_SUCCESS || !impl_->contextHandle) {
            char b[32];
            std::snprintf(b, sizeof(b), "%llx", (unsigned long long)rc);
            error = std::string("QnnContext_createFromBinary failed rc=0x") + b;
            impl_->iface().deviceFree(impl_->deviceHandle);
            impl_->iface().backendFree(impl_->backendHandle);
            impl_->deviceHandle  = nullptr;
            impl_->backendHandle = nullptr;
            return false;
        }
    }
    logI("instantiate: context created from binary (%zu bytes)", impl_->binary.size());

    // Retrieve a graph handle for each named graph we saw in inspectBinary.
    impl_->graphHandles.assign(impl_->graphs.size(), nullptr);
    for (std::size_t i = 0; i < impl_->graphs.size(); ++i) {
        const auto& g = impl_->graphs[i];
        if (impl_->iface().graphRetrieve(impl_->contextHandle,
                                         g.name.c_str(),
                                         &impl_->graphHandles[i]) != QNN_SUCCESS ||
            !impl_->graphHandles[i]) {
            error = "QnnGraph_retrieve failed for '" + g.name + "'";
            return false;
        }
        logI("instantiate: retrieved graph '%s' (in=%zu out=%zu)",
             g.name.c_str(), g.inputs.size(), g.outputs.size());
    }

    impl_->instantiated = true;
    return true;
}

namespace {

// Patch a single Qnn_Tensor_t template in-place to point at a client buffer.
// Returns true on success; false (with error) if the template version is unrecognised.
bool prepareTensorForExecute(Qnn_Tensor_t&       t,
                             Qnn_TensorType_t    role,
                             void*               data,
                             uint32_t            dataSize,
                             std::string&        error) {
    if (t.version == QNN_TENSOR_VERSION_1) {
        t.v1.type                = role;
        t.v1.memType             = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data      = data;
        t.v1.clientBuf.dataSize  = dataSize;
        return true;
    }
    if (t.version == QNN_TENSOR_VERSION_2) {
        t.v2.type                = role;
        t.v2.memType             = QNN_TENSORMEMTYPE_RAW;
        t.v2.clientBuf.data      = data;
        t.v2.clientBuf.dataSize  = dataSize;
        return true;
    }
    error = "unknown Qnn_Tensor_t version " + std::to_string(t.version);
    return false;
}

}  // namespace

bool QnnSession::execute(std::size_t                              graphIndex,
                         const std::vector<std::vector<uint8_t>>& inputs,
                         std::vector<std::vector<uint8_t>>&       outputs,
                         std::string&                             error) {
    if (!impl_->instantiated) {
        error = "execute: instantiate() must succeed first";
        return false;
    }
    if (graphIndex >= impl_->graphs.size()) {
        error = "execute: graphIndex out of range";
        return false;
    }
    const auto& g          = impl_->graphs[graphIndex];
    const uint32_t numIn   = impl_->graphNumInputs[graphIndex];
    const uint32_t numOut  = impl_->graphNumOutputs[graphIndex];

    if (inputs.size() != numIn) {
        error = "execute: input count mismatch (got " + std::to_string(inputs.size()) +
                " expected " + std::to_string(numIn) + ")";
        return false;
    }
    if (outputs.size() != numOut) {
        error = "execute: output count mismatch (got " + std::to_string(outputs.size()) +
                " expected " + std::to_string(numOut) + ")";
        return false;
    }

    // Struct-copy the templates so we don't mutate binaryInfo's memory.
    std::vector<Qnn_Tensor_t> inTensors(numIn);
    std::vector<Qnn_Tensor_t> outTensors(numOut);

    Qnn_Tensor_t* inTpl  = impl_->graphInputTemplates[graphIndex];
    Qnn_Tensor_t* outTpl = impl_->graphOutputTemplates[graphIndex];

    for (uint32_t i = 0; i < numIn; ++i) {
        const std::size_t expected = tensorByteSize(g.inputs[i]);
        if (inputs[i].size() != expected) {
            error = "execute: input[" + std::to_string(i) + "] '" + g.inputs[i].name +
                    "' size mismatch: got " + std::to_string(inputs[i].size()) +
                    " expected " + std::to_string(expected);
            return false;
        }
        inTensors[i] = inTpl[i];
        if (!prepareTensorForExecute(inTensors[i],
                                     QNN_TENSOR_TYPE_APP_WRITE,
                                     const_cast<uint8_t*>(inputs[i].data()),
                                     static_cast<uint32_t>(inputs[i].size()),
                                     error)) {
            return false;
        }
    }
    for (uint32_t i = 0; i < numOut; ++i) {
        const std::size_t expected = tensorByteSize(g.outputs[i]);
        if (outputs[i].size() != expected) {
            error = "execute: output[" + std::to_string(i) + "] '" + g.outputs[i].name +
                    "' size mismatch: got " + std::to_string(outputs[i].size()) +
                    " expected " + std::to_string(expected);
            return false;
        }
        outTensors[i] = outTpl[i];
        if (!prepareTensorForExecute(outTensors[i],
                                     QNN_TENSOR_TYPE_APP_READ,
                                     outputs[i].data(),
                                     static_cast<uint32_t>(outputs[i].size()),
                                     error)) {
            return false;
        }
    }

    auto rc = impl_->iface().graphExecute(impl_->graphHandles[graphIndex],
                                          inTensors.data(),  numIn,
                                          outTensors.data(), numOut,
                                          /*profile=*/nullptr,
                                          /*signal=*/nullptr);
    if (rc != QNN_SUCCESS) {
        error = "QnnGraph_execute failed rc=" + std::to_string(rc);
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// One-shot inspect helper used by the LocalAiApp probe.
// -----------------------------------------------------------------------------

namespace {

const char* dataTypeName(uint32_t dt) {
    switch (dt) {
        case QNN_DATATYPE_INT_8:           return "int8";
        case QNN_DATATYPE_INT_16:          return "int16";
        case QNN_DATATYPE_INT_32:          return "int32";
        case QNN_DATATYPE_INT_64:          return "int64";
        case QNN_DATATYPE_UINT_8:          return "uint8";
        case QNN_DATATYPE_UINT_16:         return "uint16";
        case QNN_DATATYPE_UINT_32:         return "uint32";
        case QNN_DATATYPE_UINT_64:         return "uint64";
        case QNN_DATATYPE_FLOAT_16:        return "float16";
        case QNN_DATATYPE_FLOAT_32:        return "float32";
        case QNN_DATATYPE_FLOAT_64:        return "float64";
        case QNN_DATATYPE_SFIXED_POINT_8:  return "sfp8";
        case QNN_DATATYPE_SFIXED_POINT_16: return "sfp16";
        case QNN_DATATYPE_SFIXED_POINT_32: return "sfp32";
        case QNN_DATATYPE_UFIXED_POINT_8:  return "ufp8";
        case QNN_DATATYPE_UFIXED_POINT_16: return "ufp16";
        case QNN_DATATYPE_UFIXED_POINT_32: return "ufp32";
        case QNN_DATATYPE_BOOL_8:          return "bool8";
        default:                           return "?";
    }
}

void appendTensor(std::ostringstream& os, const char* role, const QnnTensorInfo& t) {
    os << "    " << role << " '" << t.name << "' " << dataTypeName(t.dataType) << "[";
    for (std::size_t i = 0; i < t.dims.size(); ++i) {
        if (i) os << 'x';
        os << t.dims[i];
    }
    os << "]\n";
}

}  // namespace

std::string inspectQnnBinaryReport(const std::string& path) {
    QnnSession sess;
    std::string err;
    if (!sess.initialize(err)) return "ERROR: init: " + err;
    if (!sess.inspectBinary(path, err)) return "ERROR: inspect: " + err;

    std::ostringstream os;
    os << "QNN interface: " << sess.interfaceVersion() << "\n";
    os << "binary: " << path << "\n";
    os << "graphs: " << sess.graphs().size() << "\n";
    for (const auto& g : sess.graphs()) {
        os << "  graph '" << g.name << "'\n";
        for (const auto& t : g.inputs)  appendTensor(os, "in ", t);
        for (const auto& t : g.outputs) appendTensor(os, "out", t);
    }
    return os.str();
}

std::string probeQnnBinaryLoadReport(const std::string& path) {
    QnnSession sess;
    std::string err;
    std::ostringstream os;
    os << "QNN probe: " << path << "\n";

    if (!sess.initialize(err)) {
        os << "init: FAIL " << err << "\n";
        return os.str();
    }
    os << "init: OK (" << sess.interfaceVersion() << ")\n";

    if (!sess.inspectBinary(path, err)) {
        os << "inspect: FAIL " << err << "\n";
        return os.str();
    }
    os << "inspect: OK (" << sess.graphs().size() << " graph"
       << (sess.graphs().size() == 1 ? "" : "s") << ")\n";
    for (const auto& g : sess.graphs()) {
        os << "  graph '" << g.name << "'\n";
        for (const auto& t : g.inputs)  appendTensor(os, "in ", t);
        for (const auto& t : g.outputs) appendTensor(os, "out", t);
    }

    // The decisive step. instantiate() calls QnnContext_createFromBinary
    // against the live HTP backend — failures here surface as rc=0x... and
    // mean the binary's compile target is incompatible with the silicon.
    if (!sess.instantiate(err)) {
        os << "instantiate: FAIL " << err << "\n";
        os << "VERDICT: bundle does NOT load on this device\n";
        return os.str();
    }
    os << "instantiate: OK\n";
    os << "VERDICT: bundle loads on this device — runtime-compatible\n";
    return os.str();
}

#endif  // IMAGEGEN_HAS_QNN

}  // namespace imagegen
