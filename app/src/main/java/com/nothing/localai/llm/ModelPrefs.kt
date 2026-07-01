package com.nothing.localai.llm

import android.content.Context

/**
 * Persists which multimodal model the inference engine should load. Written and
 * read only from the `:inference` process (via [LlmRunner] / the binder's
 * setActiveModel), so a plain private SharedPreferences is safe — no
 * cross-process access. StatusActivity changes the selection over the binder,
 * never by touching this file directly.
 */
object ModelPrefs {
    private const val FILE = "model_prefs"
    private const val KEY_ACTIVE = "active_model_id"

    private fun prefs(ctx: Context) =
        ctx.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    /** The persisted selection, falling back to [ModelId.DEFAULT] (and to
     *  DEFAULT if a stale id no longer exists in the catalog). */
    fun getActiveModelId(ctx: Context): String {
        val id = prefs(ctx).getString(KEY_ACTIVE, null) ?: ModelId.DEFAULT
        return if (ModelCatalog.byId(id) != null) id else ModelId.DEFAULT
    }

    fun setActiveModelId(ctx: Context, modelId: String) {
        require(ModelCatalog.byId(modelId) != null) { "unknown model $modelId" }
        prefs(ctx).edit().putString(KEY_ACTIVE, modelId).apply()
    }
}
