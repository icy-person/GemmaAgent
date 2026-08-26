package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.desktop.WebResearchTool
import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.JvmRagStore
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.RagPlugin
import java.io.File

/** Default desktop research/knowledge bundle. Individual tools remain separately described and permission-scoped. */
class WebResearchPlugin : AgentPlugin {
    private val ragPlugin by lazy {
        RagPlugin(
            JvmRagStore(
                File(System.getProperty("user.home"), ".gemmaagent/rag/index.json")
            )
        )
    }
    private val githubPlugin by lazy { GitHubPlugin() }

    override val manifest = PluginManifest(
        id = "builtin.research-bundle",
        name = "Research & Knowledge",
        version = "2.0.0",
        description = "JavaScript-capable web research, persistent local RAG, and public GitHub tools.",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.NETWORK),
    )

    override fun tools() = buildList {
        add(WebResearchTool())
        addAll(ragPlugin.tools())
        addAll(githubPlugin.tools())
    }
}
