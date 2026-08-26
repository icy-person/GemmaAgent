package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.AgentTaskScheduler
import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.ScheduledTask
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.charset.StandardCharsets

private val runtimeJson = Json { ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }

class SchedulerPlugin : AgentPlugin {
    private val scheduler = AgentTaskScheduler()
    override val manifest = PluginManifest(
        id = "builtin.scheduler",
        name = "Task Scheduler",
        version = "1.0.0",
        description = "Create, inspect and cancel recurring agent tasks.",
        permissions = setOf(Permission.READ, Permission.WRITE),
    )
    override fun tools(): List<AgentTool> = listOf(SchedulerTool(scheduler))
}

private class SchedulerTool(private val scheduler: AgentTaskScheduler) : AgentTool {
    override val definition = ToolDefinition("scheduler", "Schedule or cancel recurring tasks. Actions: list, cancel, schedule. Execution requires an external Agent runner binding.", "agent", setOf(Permission.READ, Permission.WRITE), dangerous = false)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = runtimeJson.parseToJsonElement(argumentsJson).jsonObject
        when (o["action"]?.jsonPrimitive?.content ?: "list") {
            "list" -> ToolResult(true, "active=${scheduler.activeCount()}")
            "cancel" -> { scheduler.cancel(o["id"]?.jsonPrimitive?.content ?: error("id required")); ToolResult(true, "cancelled") }
            "schedule" -> {
                val id = o["id"]?.jsonPrimitive?.content ?: error("id required")
                val prompt = o["prompt"]?.jsonPrimitive?.content ?: error("prompt required")
                val interval = o["interval_ms"]?.jsonPrimitive?.content?.toLongOrNull() ?: error("interval_ms required")
                scheduler.schedule(ScheduledTask(id, prompt, interval, true)) { /* host binds execution in application layer */ }
                ToolResult(true, "scheduled=$id", metadata = mapOf("intervalMs" to interval.toString()))
            }
            else -> ToolResult(false, "Unknown scheduler action")
        }
    }.getOrElse { ToolResult(false, "scheduler error: ${it.message}") }
}

@Serializable
data class GraphNode(val id: String, val label: String, val type: String = "entity", val metadata: Map<String, String> = emptyMap())
@Serializable
data class GraphEdge(val from: String, val to: String, val label: String)
@Serializable
data class GraphState(val nodes: List<GraphNode> = emptyList(), val edges: List<GraphEdge> = emptyList())

class KnowledgeGraphPlugin(private val file: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.memory-graph",
        name = "Knowledge Graph",
        version = "1.0.0",
        description = "Persistent nodes and relationships for project/entity memory.",
        permissions = setOf(Permission.READ, Permission.WRITE),
    )
    override fun tools(): List<AgentTool> = listOf(KnowledgeGraphTool(file))
}

private class KnowledgeGraphTool(private val file: File) : AgentTool {
    override val definition = ToolDefinition("knowledge_graph", "Store/query graph nodes and relationships. Actions: upsert_node, link, find, dump, clear.", "memory", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = runtimeJson.parseToJsonElement(argumentsJson).jsonObject
        val state = load()
        when (o["action"]?.jsonPrimitive?.content ?: "dump") {
            "upsert_node" -> {
                val id = o["id"]?.jsonPrimitive?.content ?: error("id required")
                val node = GraphNode(id, o["label"]?.jsonPrimitive?.content ?: id, o["type"]?.jsonPrimitive?.content ?: "entity")
                save(state.copy(nodes = state.nodes.filterNot { it.id == id } + node))
                ToolResult(true, "node=$id")
            }
            "link" -> {
                val from = o["from"]?.jsonPrimitive?.content ?: error("from required")
                val to = o["to"]?.jsonPrimitive?.content ?: error("to required")
                val label = o["label"]?.jsonPrimitive?.content ?: error("label required")
                require(state.nodes.any { it.id == from } && state.nodes.any { it.id == to }) { "Both nodes must exist" }
                val edge = GraphEdge(from, to, label)
                save(state.copy(edges = (state.edges + edge).distinct()))
                ToolResult(true, "edge=${from}-$label->$to")
            }
            "find" -> {
                val q = o["query"]?.jsonPrimitive?.content?.lowercase() ?: ""
                val nodes = state.nodes.filter { it.id.lowercase().contains(q) || it.label.lowercase().contains(q) }.take(100)
                val edges = state.edges.filter { edge -> nodes.any { it.id == edge.from || it.id == edge.to } }
                ToolResult(true, buildString {
                    nodes.forEach { appendLine("NODE ${it.id}: ${it.label} [${it.type}]") }
                    edges.forEach { appendLine("EDGE ${it.from} -${it.label}-> ${it.to}") }
                })
            }
            "clear" -> { save(GraphState()); ToolResult(true, "graph cleared") }
            "dump" -> ToolResult(true, runtimeJson.encodeToString(GraphState.serializer(), state))
            else -> ToolResult(false, "Unknown graph action")
        }
    }.getOrElse { ToolResult(false, "knowledge graph error: ${it.message}") }

    private fun load(): GraphState = runCatching { if (file.isFile) runtimeJson.decodeFromString<GraphState>(file.readText()) else GraphState() }.getOrDefault(GraphState())
    private fun save(state: GraphState) { file.parentFile?.mkdirs(); val tmp = File(file.parentFile ?: file.absoluteFile.parentFile!!, file.name + ".part"); tmp.writeText(runtimeJson.encodeToString(GraphState.serializer(), state)); require(tmp.renameTo(file)) }
}

class VerificationPlugin : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.verification",
        name = "Verification",
        version = "1.0.0",
        description = "Deterministic checks for strings, files and URLs before the final answer.",
        permissions = setOf(Permission.READ),
    )
    override fun tools(): List<AgentTool> = listOf(VerificationTool())
}

private class VerificationTool : AgentTool {
    override val definition = ToolDefinition("verify", "Verify a claim using deterministic checks. Actions: contains, file_exists, url_shape.", "verification", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = runtimeJson.parseToJsonElement(argumentsJson).jsonObject
        when (o["action"]?.jsonPrimitive?.content ?: "contains") {
            "contains" -> {
                val text = o["text"]?.jsonPrimitive?.content ?: ""
                val needle = o["needle"]?.jsonPrimitive?.content ?: error("needle required")
                ToolResult(text.contains(needle, ignoreCase = false), "contains=${text.contains(needle)}")
            }
            "file_exists" -> {
                val path = File(o["path"]?.jsonPrimitive?.content ?: error("path required"))
                ToolResult(path.exists(), "exists=${path.exists()} path=${path.absolutePath}")
            }
            "url_shape" -> {
                val url = o["url"]?.jsonPrimitive?.content ?: error("url required")
                val ok = url.startsWith("https://") || url.startsWith("http://")
                ToolResult(ok, "http_url=$ok")
            }
            else -> ToolResult(false, "Unknown verification action")
        }
    }.getOrElse { ToolResult(false, "verify error: ${it.message}") }
}

class RuntimeSupportPlugin(private val dataDir: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.runtime-support",
        name = "Runtime Support",
        version = "1.0.0",
        description = "Scheduler, knowledge graph and deterministic verification.",
        permissions = setOf(Permission.READ, Permission.WRITE),
    )
    override fun tools(): List<AgentTool> = buildList {
        addAll(SchedulerPlugin().tools())
        addAll(KnowledgeGraphPlugin(File(dataDir, "memory/graph.json")).tools())
        addAll(VerificationPlugin().tools())
    }
}
