#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imagegen {

// Description of one I/O tensor in a QNN graph. Returned by QnnSession metadata
// queries — used at Phase 4a for "what does this binary expect?" debugging, and
// at Phase 6 for allocating input/output buffers in the diffusion loop.
struct QnnTensorInfo {
    std::string           name;
    uint32_t              dataType = 0;   // Qnn_DataType_t numeric value
    std::vector<uint32_t> dims;            // shape (most-major first)
    // Per-tensor scale-offset quantization (only encoding we handle for now).
    // float_value = (quantized + offset) * scale. Both 0 → tensor is not
    // quantized (or uses an encoding we don't surface).
    float                 quantScale  = 0.0f;
    int32_t               quantOffset = 0;
};

struct QnnGraphInfo {
    std::string                name;
    std::vector<QnnTensorInfo> inputs;
    std::vector<QnnTensorInfo> outputs;
};

// Bytes per element of a Qnn_DataType_t value. Returns 0 for unknown/sub-byte
// types — caller should treat that as "unsupported" rather than buffer-sizing.
std::size_t bytesPerDataType(uint32_t dataType);

// Total byte size of a tensor host buffer (elementCount * bytesPerElement).
std::size_t tensorByteSize(const QnnTensorInfo& info);

// RAII wrapper around a single QNN HTP context — owns the backend, device, and
// (optionally) a context binary loaded from disk. Compiled out when the build
// is configured without QAIRT (IMAGEGEN_HAS_QNN not defined); the methods then
// return false with an explanatory error.
class QnnSession {
public:
    QnnSession();
    ~QnnSession();

    QnnSession(const QnnSession&)            = delete;
    QnnSession& operator=(const QnnSession&) = delete;

    // dlopen libQnnHtp.so + libQnnSystem.so, resolve interfaces, create
    // log/backend/device. Idempotent: calling twice has no effect.
    bool initialize(std::string& error);

    // Read the file at `path` and use QnnSystem to inspect its metadata
    // (graph names + tensor shapes). Does NOT construct a runnable context yet
    // — that is `instantiate()`. Cheap; safe to call early to "preview" a model.
    bool inspectBinary(const std::string& path, std::string& error);

    // After a successful inspectBinary, materialize the context so the graphs
    // can be executed. Creates a real HTP backend + runtime context from the
    // cached binary, retrieves a handle for every graph the binary advertises.
    bool instantiate(std::string& error);

    // Run a single forward pass on the graph at `graphIndex`.
    //
    // `inputs` and `outputs` are *raw byte buffers*. They MUST exactly match
    // the sizes implied by the graph's tensor metadata (use tensorByteSize on
    // each entry of graphs()[graphIndex].inputs / .outputs). Caller is
    // responsible for any fp32↔fp16/quant packing.
    //
    // The output buffers must be pre-sized; QnnSession does NOT resize them.
    bool execute(std::size_t                            graphIndex,
                 const std::vector<std::vector<uint8_t>>& inputs,
                 std::vector<std::vector<uint8_t>>&       outputs,
                 std::string&                             error);

    // Whether initialize() has completed.
    bool initialized() const;

    // Whether instantiate() has completed.
    bool instantiated() const;

    // Graph metadata, valid after inspectBinary().
    const std::vector<QnnGraphInfo>& graphs() const;

    // Human-readable single-line summary of the QNN core API version detected
    // from libQnnHtp.so during initialize(). Empty if not initialized.
    std::string interfaceVersion() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience: full one-shot inspect. Initializes a session, runs inspectBinary,
// formats a multi-line report. Returns the report (or an "ERROR: ..." line).
std::string inspectQnnBinaryReport(const std::string& path);

// Compat probe: init + inspectBinary + instantiate against the running device.
// inspectBinary is metadata-only and parses; instantiate is the call that
// actually hands the binary to libQnnHtp and surfaces failures like
// rc=0x138d when the binary's HTP target arch is incompatible with the
// silicon (e.g. an _8gen3.zip / V75-targeted bundle on a V73 device).
//
// The report has one line per stage with a clear PASS/FAIL marker and the
// QNN rc on failure, suitable for a yes/no answer to "does this bundle load
// on this device?" without touching the rest of the bundle.
std::string probeQnnBinaryLoadReport(const std::string& path);

}  // namespace imagegen
