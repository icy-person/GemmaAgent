package com.example.gemmaagent

import android.app.Application
import com.example.gemmaagent.shared.AndroidAgentContext

class GemmaAgentApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        AndroidAgentContext.context = applicationContext
    }
}
