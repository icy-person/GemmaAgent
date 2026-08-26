package com.example.gemmaagent.shared

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class PluginManifest(
    val id: String,
    val name: String,
    val version: String,
    val description: String = "",
    val permissions: Set<Permission> = setOf(Permission.READ),
    val tools: List<ToolDefinition> = emptyList(),
)

interface AgentPlugin {
    val manifest: PluginManifest
    fun tools(): List<AgentTool>
}

class PluginRegistry {
    private val plugins = linkedMapOf<String, AgentPlugin>()
    private val toolOwners = mutableMapOf<String, String>()

    fun install(plugin: AgentPlugin) {
        require(plugin.manifest.id.isNotBlank()) { "Plugin id is required" }
        require(plugins[plugin.manifest.id] == null) {
            "Plugin already installed: ${plugin.manifest.id}"
        }

        for (registeredTool in plugin.tools()) {
            val definition = registeredTool.definition
            require(toolOwners[definition.name] == null) {
                "Tool already registered: ${definition.name}"
            }
            require(definition.permissions.all { permission ->
                plugin.manifest.permissions.contains(permission)
            }) {
                "Plugin ${plugin.manifest.id} requests an undeclared tool permission"
            }
            toolOwners[definition.name] = plugin.manifest.id
        }

        plugins[plugin.manifest.id] = plugin
    }

    fun uninstall(id: String) {
        val plugin = plugins.remove(id) ?: return
        for (registeredTool in plugin.tools()) {
            toolOwners.remove(registeredTool.definition.name)
        }
    }

    fun all(): List<AgentPlugin> = plugins.values.toList()

    fun findToolOwner(tool: String): String? = toolOwners[tool]

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        fun parseManifest(text: String): PluginManifest =
            json.decodeFromString(text)
    }
}
