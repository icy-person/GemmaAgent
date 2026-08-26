package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.desktop.WebResearchTool
import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.Permission

class WebResearchPlugin : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.web-research",
        name = "Web Research",
        version = "1.0.0",
        description = "JavaScript-capable web research engine with search, page extraction, ranking, deduplication and source evidence.",
        permissions = setOf(Permission.NETWORK),
    )

    override fun tools() = listOf(WebResearchTool())
}
