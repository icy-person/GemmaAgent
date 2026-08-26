package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.JvmRagStore
import com.example.gemmaagent.shared.PluginRegistry
import com.example.gemmaagent.shared.RagPlugin
import java.io.File

suspend fun installDefaultPlugins(registry: PluginRegistry, dataDir: File) {
    registry.install(WebResearchPlugin())
    registry.install(GitHubPlugin())
    registry.install(RagPlugin(JvmRagStore(File(dataDir, "rag/index.json"))))
}
