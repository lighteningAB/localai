package com.nothing.localai.ui

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.nothing.localai.R
import com.nothing.localai.llm.LlmDownloader
import com.nothing.localai.llm.ModelCatalog
import com.nothing.localai.llm.ModelId

class StatusActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_status)
        val tv = findViewById<TextView>(R.id.status_text)
        val spec = ModelCatalog.byId(ModelId.DEFAULT)
        val s = LlmDownloader(applicationContext).statusOf(ModelId.DEFAULT)
        val label = spec?.fileName ?: ModelId.DEFAULT
        tv.text = "$label: ${s.state} (${s.bytesDownloaded}/${s.totalBytes})"
    }
}
