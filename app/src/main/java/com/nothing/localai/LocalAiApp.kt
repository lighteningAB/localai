package com.nothing.localai

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager

class LocalAiApp : Application() {
    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Local AI",
                NotificationManager.IMPORTANCE_MIN
            )
        )
    }

    companion object {
        const val CHANNEL_ID = "localai_inference"
    }
}
