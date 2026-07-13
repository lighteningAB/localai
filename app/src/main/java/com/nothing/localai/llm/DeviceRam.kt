package com.nothing.localai.llm

import android.app.ActivityManager
import android.content.Context

/** Device RAM introspection for the [ModelSpec.minRamBytes] gate. */
object DeviceRam {
    fun totalBytes(ctx: Context): Long {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        return mi.totalMem
    }

    /** True if the device has enough RAM to attempt [spec] (floor of 0 = always). */
    fun meetsFloor(ctx: Context, spec: ModelSpec): Boolean =
        spec.minRamBytes <= 0L || totalBytes(ctx) >= spec.minRamBytes
}
