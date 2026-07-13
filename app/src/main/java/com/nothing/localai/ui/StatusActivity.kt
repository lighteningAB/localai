package com.nothing.localai.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.nothing.localai.ILocalAiService
import com.nothing.localai.ITokenCallback
import com.nothing.localai.R
import com.nothing.localai.llm.Accelerator
import com.nothing.localai.llm.LlmDownloader
import com.nothing.localai.llm.ModelCatalog
import com.nothing.localai.llm.ModelId
import com.nothing.localai.llm.ModelSpec
import java.util.concurrent.Executors

private const val TAG = "StatusActivity"

class StatusActivity : AppCompatActivity() {

    private val io = Executors.newSingleThreadExecutor()
    private val downloader by lazy { LlmDownloader(applicationContext) }

    private lateinit var statusText: TextView
    private lateinit var group: RadioGroup
    private lateinit var note: TextView
    private lateinit var testButton: Button
    private lateinit var testOutput: TextView

    // viewId -> modelId, so the RadioGroup check listener can resolve selections.
    private val buttonModel = mutableMapOf<Int, String>()

    // Set by `--es seammodel <id>`: switch active model over the binder then run
    // the diagnostic through the service (exercises EngineManager + the backend
    // seam end-to-end, not just NativeLlama). Consumed once the service binds.
    @Volatile private var pendingSeamModel: String? = null

    // Single-flight guard for the llmtest smoke (see maybeRunLlamaSmoke).
    private val smokeRunning = java.util.concurrent.atomic.AtomicBoolean(false)

    @Volatile private var service: ILocalAiService? = null
    // Suppresses the check listener while we programmatically sync the UI to the
    // service's active model, so syncing doesn't fire a spurious setActiveModel.
    private var syncing = false

    private val conn = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = ILocalAiService.Stub.asInterface(binder)
            refreshActiveModel()
            pendingSeamModel?.let { m -> pendingSeamModel = null; runSeamTest(m) }
        }
        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_status)
        statusText = findViewById(R.id.status_text)
        group = findViewById(R.id.model_group)
        note = findViewById(R.id.model_note)
        testButton = findViewById(R.id.test_button)
        testOutput = findViewById(R.id.test_output)

        buildRadioButtons()
        group.setOnCheckedChangeListener { _, checkedId ->
            if (syncing) return@setOnCheckedChangeListener
            val modelId = buttonModel[checkedId] ?: return@setOnCheckedChangeListener
            applyModel(modelId)
        }
        testButton.setOnClickListener { runDiagnostic() }
        renderStatus(ModelId.DEFAULT)
        maybeRunLlamaSmoke(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        maybeRunLlamaSmoke(intent)
    }

    /**
     * Throwaway bring-up harness for the llama.cpp engine, triggered by
     *   adb shell am start -n com.nothing.localai.debug/com.nothing.localai.ui.StatusActivity \
     *     --es llmtest "your prompt"
     * Loads libllmcpp, pings, and (if the 12B QAT GGUF is present in files/models)
     * loads it on the CPU and streams one turn — logging tokens/sec to logcat
     * ("LlamaSmoke"). Remove once the engine seam lands.
     */
    /** Switch the active model over the binder, then run the through-service
     *  diagnostic — the real seam path (EngineManager -> LlamaCppRunner). */
    private fun runSeamTest(modelId: String) {
        val svc = service ?: return
        io.execute {
            val now = runCatching { svc.setActiveModel(modelId) }.getOrNull()
            Log.i("SeamTest", "setActiveModel($modelId) -> active now=$now")
            runOnUiThread {
                if (now != modelId) {
                    testOutput.text = "setActiveModel refused/failed: active=$now"
                }
                runDiagnostic()
            }
        }
    }

    private fun maybeRunLlamaSmoke(intent: Intent?) {
        intent?.getStringExtra("seammodel")?.let { m ->
            pendingSeamModel = m
            if (service != null) { pendingSeamModel = null; runSeamTest(m) }
            return
        }
        val prompt = intent?.getStringExtra("llmtest") ?: return
        // Optional GBNF grammar to exercise constrained decoding. `--es grammar json`
        // is a built-in shortcut (adb can't pass multi-line GBNF) forcing a
        // {"title": "...", "items": [...]} object — a UI-spec proxy.
        val grammar = when (intent.getStringExtra("grammar")) {
            "json" -> """
                root  ::= "{" ws "\"title\"" ws ":" ws str ws "," ws "\"items\"" ws ":" ws arr ws "}"
                str   ::= "\"" char* "\""
                char  ::= [^"\\]
                arr   ::= "[" ws (str (ws "," ws str)*)? ws "]"
                ws    ::= [ \t\n]*
            """.trimIndent()
            else -> intent.getStringExtra("grammar") ?: ""
        }
        io.execute {
            val T = "LlamaSmoke"
            // Guard against double-fire (onCreate + onNewIntent both consuming the
            // extra): two concurrent runs each loaded a 7 GB model — the transient
            // page-in spike drew an LMK signal-9 even on the 24 GB board.
            if (!smokeRunning.compareAndSet(false, true)) {
                Log.w(T, "smoke already running; ignoring duplicate trigger"); return@execute
            }
            try {
            if (!com.nothing.localai.llm.NativeLlama.ensureLoaded()) {
                Log.e(T, "libllmcpp.so failed to load"); return@execute
            }
            Log.i(T, "ping: ${com.nothing.localai.llm.NativeLlama.nativePing()}")
            val gguf = java.io.File(filesDir, "models/gemma-4-12b-it-qat-q4_0.gguf")
            if (!gguf.exists()) { Log.w(T, "no gguf at ${gguf.absolutePath}; ping-only"); return@execute }
            // Match the throttled LlamaCppRunner caps (2 threads, 2048 ctx) — the
            // 8-thread/4096 config cooked the board. Keep this debug path safe.
            val nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(2)
            val t0 = System.currentTimeMillis()
            val model = com.nothing.localai.llm.NativeLlama.nativeLoadModel(gguf.absolutePath)
            if (model == 0L) { Log.e(T, "load failed"); return@execute }
            Log.i(T, "model loaded in ${System.currentTimeMillis() - t0}ms")
            val ctx = com.nothing.localai.llm.NativeLlama.nativeCreateContext(model, 2048, nThreads, 42)
            if (ctx == 0L) { Log.e(T, "ctx failed"); com.nothing.localai.llm.NativeLlama.nativeFreeModel(model); return@execute }
            val nTok = intArrayOf(0)
            val g0 = System.currentTimeMillis()
            val sink = object : com.nothing.localai.llm.NativeLlama.TokenSink {
                override fun onToken(text: String) { nTok[0]++ }
            }
            if (grammar.isNotEmpty()) Log.i(T, "grammar constrained (${grammar.length} chars)")
            val out = com.nothing.localai.llm.NativeLlama.nativeGenerate(ctx, model, prompt, grammar, 128, sink)
            val dt = System.currentTimeMillis() - g0
            val tps = if (dt > 0) nTok[0] * 1000.0 / dt else 0.0
            Log.i(T, "generated ${nTok[0]} tok in ${dt}ms = %.1f tok/s".format(tps))
            Log.i(T, "OUTPUT: $out")
            com.nothing.localai.llm.NativeLlama.nativeFreeContext(ctx)
            com.nothing.localai.llm.NativeLlama.nativeFreeModel(model)
            } finally {
                smokeRunning.set(false)
            }
        }
    }

    /**
     * Dev-only: run one generate against the active model so the engine-init /
     * NPU load path is exercised on device (drive from adb: launch this
     * activity, tap the model, tap "Test active model"). Everything is logged
     * to logcat tag "NpuDiag"; the on-screen field mirrors the outcome.
     */
    private fun runDiagnostic() {
        val svc = service
        if (svc == null) {
            testOutput.text = "service not connected"
            return
        }
        testOutput.text = "loading engine + generating…"
        Log.i("NpuDiag", "diagnostic start: active=${runCatching { svc.getActiveModel() }.getOrNull()}")
        val activeId = runCatching { svc.getActiveModel() }.getOrNull()
        val dispatchAt = android.os.SystemClock.elapsedRealtime()
        val cb = object : ITokenCallback.Stub() {
            private val sb = StringBuilder()
            var firstTokenAt = 0L
            var lastTokenAt = 0L
            var tokenCount = 0
            override fun onToken(rid: String, text: String) {
                val now = android.os.SystemClock.elapsedRealtime()
                if (firstTokenAt == 0L) firstTokenAt = now
                lastTokenAt = now
                tokenCount++
                sb.append(text)
                runOnUiThread { testOutput.text = sb.toString() }
            }
            override fun onDone(rid: String, full: String) {
                val prefillMs = firstTokenAt - dispatchAt
                val decodeMs = (lastTokenAt - firstTokenAt).coerceAtLeast(1)
                // tokenCount-1 intervals between first and last streamed token
                val decodeTps = (tokenCount - 1) * 1000.0 / decodeMs
                val chars = full.length
                Log.i(
                    "NpuBench",
                    "BENCH model=$activeId streamedTokens=$tokenCount chars=$chars " +
                        "prefillMs=$prefillMs decodeMs=$decodeMs decodeTokPerSec=%.2f".format(decodeTps)
                )
                runOnUiThread {
                    testOutput.text =
                        "$activeId\ntoks=$tokenCount prefill=${prefillMs}ms " +
                        "decode=%.1f tok/s".format(decodeTps)
                }
            }
            override fun onError(rid: String, code: String, msg: String) {
                Log.e("NpuBench", "ERROR [$code]: $msg")
                runOnUiThread { testOutput.text = "ERROR [$code]: $msg" }
            }
        }
        io.execute {
            try {
                svc.createSession("npu-diag")
                val prompt = "Write a detailed 200-word product description for a new " +
                    "smartphone with a great camera and long battery life. Use vivid language."
                val rid = svc.generate("npu-diag", prompt, cb)
                Log.i("NpuBench", "generate dispatched rid=$rid model=$activeId")
            } catch (t: Throwable) {
                Log.e("NpuBench", "diagnostic threw", t)
                runOnUiThread { testOutput.text = "THREW: ${t.message}" }
            }
        }
    }

    override fun onStart() {
        super.onStart()
        // Bind the inference service (separate :inference process) to read and
        // switch the active model over the binder. Cross-process prefs aren't
        // reliable, so the binder is the source of truth.
        val intent = Intent(this, com.nothing.localai.LocalAiService::class.java)
            .setAction("com.nothing.localai.BIND")
        if (!bindService(intent, conn, Context.BIND_AUTO_CREATE)) {
            note.text = "Could not bind inference service."
        }
    }

    override fun onStop() {
        super.onStop()
        runCatching { unbindService(conn) }
        service = null
    }

    override fun onDestroy() {
        super.onDestroy()
        io.shutdownNow()
    }

    private fun buildRadioButtons() {
        ModelCatalog.selectable.forEach { spec ->
            val rb = RadioButton(this).apply {
                id = android.view.View.generateViewId()
                text = label(spec)
            }
            buttonModel[rb.id] = spec.id
            group.addView(rb)
        }
    }

    private fun label(spec: ModelSpec): String {
        val accel = when {
            spec.backend == com.nothing.localai.llm.Backend.LLAMACPP -> "llama.cpp CPU"
            spec.accelerator == Accelerator.NPU -> "Hexagon NPU"
            else -> "CPU+GPU"
        }
        return "${spec.fileName}  ·  $accel"
    }

    /** Ask the service which model is live and check the matching radio. */
    private fun refreshActiveModel() {
        val svc = service ?: return
        io.execute {
            val active = runCatching { svc.getActiveModel() }.getOrDefault(ModelId.DEFAULT)
            runOnUiThread {
                syncing = true
                val viewId = buttonModel.entries.firstOrNull { it.value == active }?.key
                if (viewId != null) group.check(viewId) else group.clearCheck()
                syncing = false
                renderStatus(active)
                note.text = ""
            }
        }
    }

    /** Tell the service to switch models; reloads lazily on the next request. */
    private fun applyModel(modelId: String) {
        val svc = service
        if (svc == null) {
            note.text = "Service not connected yet — try again."
            refreshActiveModel()
            return
        }
        val spec = ModelCatalog.byId(modelId)
        note.text = "Switching to ${spec?.fileName ?: modelId}…"
        io.execute {
            val active = runCatching { svc.setActiveModel(modelId) }
                .onFailure { Log.e(TAG, "setActiveModel failed", it) }
                .getOrDefault(modelId)
            runOnUiThread {
                renderStatus(active)
                note.text = if (active == modelId) {
                    "Active: ${spec?.fileName}. Engine reloads on the next request."
                } else {
                    "Switch rejected — still on $active."
                }
                // Re-sync the radio in case the service kept the old model.
                if (active != modelId) {
                    syncing = true
                    val viewId = buttonModel.entries.firstOrNull { it.value == active }?.key
                    if (viewId != null) group.check(viewId)
                    syncing = false
                }
            }
        }
    }

    private fun renderStatus(modelId: String) {
        val spec = ModelCatalog.byId(modelId)
        val s = downloader.statusOf(modelId)
        val labelName = spec?.fileName ?: modelId
        statusText.text = "$labelName: ${s.state} (${s.bytesDownloaded}/${s.totalBytes})"
    }
}
