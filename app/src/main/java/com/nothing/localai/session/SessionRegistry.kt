package com.nothing.localai.session

import android.content.Context
import com.nothing.localai.llm.InferenceRunner
import com.nothing.localai.llm.InferenceSession
import java.util.concurrent.ConcurrentHashMap

/**
 * One [ChatSession] per widget sessionId. LiteRT-LM 0.11.0 enforces a hard
 * limit of one [com.google.ai.edge.litertlm.Conversation] per Engine — the
 * release-notes claim of "multi-session support" turns out to be CLI-only;
 * the Android API throws FAILED_PRECONDITION on the second createConversation.
 *
 * So this registry enforces a single-active policy: creating a session for a
 * new sid closes every other live one first. Widgets can still each have
 * their own ChatSession identity, but only the most-recently-touched one
 * keeps a live Conversation. KV cache is rebuilt the next time a stale
 * widget becomes active.
 *
 * [maxLive] is retained for API compatibility but pinned to 1 in practice.
 */
class SessionRegistry(
    private val runner: InferenceRunner,
    @Suppress("unused") private val ctx: Context,
    @Suppress("unused") private val maxLive: Int = 1,
) {

    private val sessions = ConcurrentHashMap<String, InferenceSession>()

    fun getOrCreate(sessionId: String): InferenceSession {
        sessions[sessionId]?.let { return it }
        synchronized(sessions) {
            sessions[sessionId]?.let { return it }
            // Close every other live session synchronously — LiteRT-LM's
            // engine refuses createConversation while any Conversation is
            // still open against it (llama.cpp has no such limit, but keeping a
            // single live KV bounds RAM either way).
            sessions.entries.toList().forEach { (sid, s) ->
                sessions.remove(sid)
                runCatching { s.close() }
            }
            val s = runner.newSession(sessionId)
            sessions[sessionId] = s
            return s
        }
    }

    fun reset(sessionId: String) {
        sessions[sessionId]?.reset()
    }

    fun release(sessionId: String) {
        sessions.remove(sessionId)?.close()
    }

    fun cancel(requestId: String) {
        runner.cancel(requestId)
    }

    fun releaseAll() {
        sessions.values.forEach { runCatching { it.close() } }
        sessions.clear()
    }
}
