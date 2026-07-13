# Demo: Gemma 4 12B driving ai-playground cards, on-device

What to show colleagues, and exactly how. Everything below is **verified
end-to-end** on the SM8850 canoe board (adb `7ec11429`, 24 GB RAM) with real
screenshots — including the full ai-playground canvas running on the 12B.

## The one-line story

> Gemma 4 **12B** was a dead end on our stack — the LiteRT-LM GPU runtime can't
> allocate its logits tensor (Adreno 1 GB cap; Google [#2461] still open). We
> added a **second inference backend (llama.cpp, CPU)** to localai, so the same
> canvas app now runs in two gears: **E4B for fast interaction, 12B for depth**
> — plus **grammar-constrained decoding** (the model *cannot* emit invalid JSON).

[#2461]: https://github.com/google-ai-edge/LiteRT-LM/issues/2461

## The demo: two gears, one app (ai-playground canvas)

### Act 1 — fast gear (E4B, the fluid canvas)
Active model = `gemma4-e4b-it-int4`. Voice or adb-inject a few quick cards:
- "Make me a workout timer with a 3-item checklist"
- "Weather in London" / "coffee shops near Soho" (map)
- "Morning brief" (agentic ai_refresh tile — allowed on E4B)
Snappy multi-card canvas; this is the product experience.

### Act 2 — deep gear (12B, "watch it think") ⭐
Switch the model in localai's StatusActivity (or
`--es seammodel gemma4-12b-it-qat`), then back in the canvas:

- **"Plan a 3 day Tokyo food trip as sections with checklists"** (~3.5 min)
  → a genuinely impressive card: Day 1 Tsukiji & Ginza (Tsukiji Outer Market
  breakfast, Kabayaki in Ginza, Hamarikyu matcha…), Day 2 Shibuya & Shinjuku
  (Omoide Yokocho yakitori, Golden Gai…), Day 3 Asakusa & Ueno — knowledge and
  organization E4B simply cannot produce.
- Shorter filler while people ask questions: **"Make me a workout timer card
  with a 3 item checklist"** (~90 s).

**Say:** "Twelve billion parameters, airplane mode, in your hand. It's not the
fast gear — it's the *smart* gear, and the app knows the difference."

The app auto-detects the heavy model (HeavyMode) and clamps itself: no per-card
agent sessions, 2 tool hops max, tighter session rotation — that's why this is
stable now (the unconstrained first attempt hard-rebooted the board).

### Act 3 — guaranteed-valid JSON (GBNF)
```
adb -s 7ec11429 shell am force-stop com.nothing.localai.debug
adb -s 7ec11429 shell am start -n com.nothing.localai.debug/com.nothing.localai.ui.StatusActivity \
  --es llmtest "'List three fruits with a title'" --es grammar "json"
adb -s 7ec11429 logcat -s LlamaSmoke
```
Always exact `{"title": …, "items": […]}` — the grammar masks every invalid
token at sampling time. "Malformed output is impossible, not just unlikely."

### Talking point — RAM guard
`setActiveModel(12B)` on a device under 12 GB RAM is **refused** (previous
model kept, clean error): `EngineManager: setActiveModel: REFUSED … keeping …`.
No OOM roulette on phones that can't hold it.

## Setup (before colleagues arrive)

1. Build + install **localai** (`feature/llamacpp-12b`) and **ai-playground**
   (`master`). Both debug builds bind fine (BIND_AI is normal-level).
   **Order matters:** if you ever reinstall localai, reinstall ai-playground
   after it — the BIND_AI grant is re-derived at install time.
2. Models in localai's `files/models/` (push via `/data/local/tmp` + `run-as`):
   `gemma-4-12b-it-qat-q4_0.gguf` (~6.5 GB) and `gemma-4-E4B-it.litertlm`.
3. **Prewarm the 12B once** (any short llmtest/seammodel run) before the room
   is watching, and pre-generate the Tokyo card if you want it instantly
   revisitable on the canvas.

## Timing + safety facts (measured)

| | E4B | 12B (llama.cpp CPU) |
|---|---|---|
| Simple card | seconds | ~90 s |
| Rich card (Tokyo trip) | – | ~3.5 min (1351-tok prefill 79 s + 335-tok spec) |
| Decode rate | fast | ~5–6.5 tok/s (2 threads — faster than 8, which only throttled) |
| Thermals | fine | peaks ~103 °C bursts, governor holds; survived repeated runs |

Stability fixes that made this possible (committed): decode capped at 2
threads / prefill 4 (asymmetric), prefill chunked to n_batch=512 (whole-prompt
batches killed the process), ggml fused RMS_NORM+MUL kernel disabled
(`GGML_CPU_DISABLE_FUSION` — SIGSEGV on multi-threaded large-batch prefill),
HeavyMode clamps in ai-playground.

## If something goes wrong live
- Engine died: `adb shell monkey -p com.nothing.localai.debug -c android.intent.category.LAUNCHER 1`
- adb lost: `adb kill-server && adb start-server`
- Never force-stop mid-generation; never queue several 12B cards at once.
- Fallback: E4B carries the canvas demo alone + Act 3 GBNF standalone still
  tells the 12B story.
