package com.example.gemmaagent.shared

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

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
    private val mutex = Mutex()
    private val plugins = linkedMapOf<String, AgentPlugin>()
    private val toolOwners = mutableMapOf<String, String>()

    suspend fun install(plugin: AgentPlugin) = mutex.withLock {
        require(plugin.manifest.id.isNotBlank()) { "Plugin id is required" }
        require(plugins[plugin.manifest.id] == null) { "Plugin already installed: ${plugin.manifest.id}" }
        val tools = plugin.tools()
        require(tools.map { it.definition.name }.distinct().size == tools.size) { "Plugin contains duplicate tool names" }
        for (registeredTool in tools) {
            val definition = registeredTool.definition
            require(toolOwners[definition.name] == null) { "Tool already registered: ${definition.name}" }
            require(definition.permissions.all(plugin.manifest.permissions::contains)) {
                "Plugin ${plugin.manifest.id} requests an undeclared tool permission"
            }
        }
        tools.forEach { toolOwners[it.definition.name] = plugin.manifest.id }
        plugins[plugin.manifest.id] = plugin
    }

    suspend fun uninstall(id: String) = mutex.withLock {
        val plugin = plugins.remove(id) ?: return@withLock
        plugin.tools().forEach { toolOwners.remove(it.definition.name) }
    }

    suspend fun all(): List<AgentPlugin> = mutex.withLock { plugins.values.toList() }

    suspend fun allTools(): List<AgentTool> = mutex.withLock { plugins.values.flatMap { it.tools() } }

    suspend fun findToolOwner(tool: String): String? = mutex.withLock { toolOwners[tool] }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }
        fun parseManifest(text: String): PluginManifest = json.decodeFromString(text)
    }
}
