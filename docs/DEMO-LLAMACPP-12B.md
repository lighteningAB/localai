# Demo: Gemma 4 12B on-device + constrained decoding

What to show colleagues, and exactly how. Everything here is verified on the
SM8850 **canoe** board (adb serial `7ec11429`, 24 GB RAM).

## The one-line story

> Gemma 4 **12B** was a dead end on our LiteRT-LM stack — the GPU runtime can't
> allocate its logits tensor (Adreno 1 GB cap; Google issue [#2461] is still
> open). We added a **second inference backend (llama.cpp, CPU)** so the 12B QAT
> model runs on-device, **and** it gives us **grammar-constrained decoding** —
> the model is now *incapable* of emitting invalid UI-spec JSON.

[#2461]: https://github.com/google-ai-edge/LiteRT-LM/issues/2461

---

## One-time setup (do before colleagues arrive)

1. **On office network / VPN** (the APK is signed by `sign.nothing.local`):
   ```
   cd ~/Documents/github/localai
   git checkout feature/llamacpp-12b
   git submodule update --init app/src/main/cpp/3rdparty/llama.cpp
   ./gradlew :app:assembleDebug
   adb -s 7ec11429 install -r app/build/outputs/apk/debug/app-debug.apk
   ```
   First build compiles llama.cpp (~8 min); later builds are cached.

2. **Push the model** (~7 GB, one-time; survives reinstalls of the *same* signer):
   ```
   MODEL=gemma-4-12b-it-qat-q4_0.gguf   # google/gemma-4-12B-it-qat-q4_0-gguf on HF
   adb -s 7ec11429 push $MODEL /data/local/tmp/
   adb -s 7ec11429 shell "run-as com.nothing.localai.debug cp /data/local/tmp/$MODEL files/models/"
   adb -s 7ec11429 shell rm /data/local/tmp/$MODEL
   ```
   (Keep a Gemma-4 E4B `.litertlm` in `files/models/` too, for the switch demo.)

3. **Warm it once** before the room is watching — first load mmaps 7 GB.

**Expectations to set:** 12B on CPU is **~5 tokens/sec** (deliberate, readable
speed — not a chatbot race). Load is ~3 s. Keep demo prompts short.

---

## Demo A — 12B thinking entirely on-device (GUI, no laptop needed)

The cleanest live demo. Open the **Local AI** status screen on the phone:

1. It lists selectable models. Tap **`gemma-4-12b-it-qat-q4_0.gguf · llama.cpp CPU`**.
2. Tap **Test**. Watch tokens stream in; the footer shows `decode = ~5 tok/s`.

**Say:** "This is a 12-billion-parameter model, no network, running on the CPU.
The same model refuses to load on the GPU runtime — this is the only way it runs
on the phone at all." Airplane-mode the device first for effect.

Equivalent over adb (if projecting a terminal):
```
adb -s 7ec11429 shell am start -n com.nothing.localai.debug/com.nothing.localai.ui.StatusActivity \
  --es seammodel "gemma4-12b-it-qat"
adb -s 7ec11429 logcat -s NpuBench   # look for the BENCH line: streamedTokens / tok/s
```

---

## Demo B — the model *cannot* produce invalid JSON (GBNF) ⭐

The headline. This is the reliability win for AI-generated UI.

```
adb -s 7ec11429 shell am force-stop com.nothing.localai.debug
adb -s 7ec11429 shell am start -n com.nothing.localai.debug/com.nothing.localai.ui.StatusActivity \
  --es llmtest "'List three fruits with a title'" --es grammar "json"
adb -s 7ec11429 logcat -s LlamaSmoke
```

Output is **guaranteed** to match the grammar, e.g.:
```json
{ "title": "Fresh Summer Fruits", "items": ["Mango", "Watermelon", "Pineapple"] }
```

**Say:** "The grammar constrains the sampler at every token — malformed JSON is
mathematically impossible, not just unlikely. Today the app hand-repairs broken
model output; with this, the composer can't emit a broken UI spec in the first
place." (Optionally run it a few times — always valid.)

---

## Demo C — graceful on low-RAM devices (safety)

We never OOM-kill a phone that can't hold the 12B. The service checks device RAM
against a 12 GB floor and refuses, keeping the current model.

Talking point + log evidence (canoe passes the floor, so this is the log from a
simulated high floor during bring-up):
```
EngineManager: setActiveModel: REFUSED gemma4-12b-it-qat — device RAM ... < floor ...; keeping <previous>
```
**Say:** "On an 8 GB phone the switch is refused and the caller is told, instead
of the process getting killed mid-generation."

---

## Demo D — next step (not yet wired end-to-end)

Wiring the 12B as the **ai-playground composer brain** (voice → 12B → UI cards),
and having ai-playground call the new `generateConstrained()` so every card spec
is grammar-valid. The localai side is done and exposed over AIDL (slot 20); the
ai-playground binder mirror + a "use 12B" toggle are the remaining glue. Mention
as "the payoff this unlocks," not something to click today.

---

## If something goes wrong live

- **Engine/service died** (`DeadObjectException`): relaunch —
  `adb -s 7ec11429 shell monkey -p com.nothing.localai.debug -c android.intent.category.LAUNCHER 1`.
- **Model not found**: re-check `run-as com.nothing.localai.debug ls -l files/models/`.
- **adb "device not found"**: `adb kill-server && adb start-server`.
- **Don't** force-stop mid-generation; **don't** expect chatbot speed — 5 tok/s is
  the story (a 12B model, on a CPU, in your hand).

## What's committed

Branch `feature/llamacpp-12b`. New backend seam (`InferenceRunner`/`Session`,
`EngineManager`), `LlamaCppRunner`/`LlamaSession`, native `libllmcpp.so`
(llama.cpp submodule), `NativeLlama` JNI, RAM guard (`ModelSpec.minRamBytes`),
and `generateConstrained()` AIDL. Existing LiteRT path and AIDL calls unchanged.
