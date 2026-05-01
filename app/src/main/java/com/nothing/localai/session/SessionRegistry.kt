package com.nothing.localai.session

import android.content.Context
import com.nothing.localai.llm.ChatSession
import com.nothing.localai.llm.LlmRunner
import java.util.concurrent.ConcurrentHashMap

/**
 * One [ChatSession] per widget sessionId. Bounded with a soft cap to keep KV
 * cache memory under control. With MediaPipe LlmInferenceSession, the
 * underlying engine internally serializes — but we still cap concurrent
 * sessions because each has its own KV cache.
 */
class SessionRegistry(
    private val runner: LlmRunner,
    private val ctx: Context,
    private val maxLive: Int = 4,
) {

    private val sessions = ConcurrentHashMap<String, ChatSession>()

    fun getOrCreate(sessionId: String): ChatSession {
        sessions[sessionId]?.let { return it }
        synchronized(sessions) {
            sessions[sessionId]?.let { return it }
            if (sessions.size >= maxLive) evictOne()
            val s = ChatSession(sessionId, runner, ctx)
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

    private fun evictOne() {
        val (id, s) = sessions.entries.firstOrNull() ?: return
        sessions.remove(id)
        runCatching { s.close() }
    }
}
