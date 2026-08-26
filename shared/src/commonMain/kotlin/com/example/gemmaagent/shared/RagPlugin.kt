package com.example.gemmaagent.shared

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class RagPlugin(private val store: RagStore) : AgentPlugin {
    private val engine = RagEngine(store)
    private val json = Json { ignoreUnknownKeys = true }

    override val manifest = PluginManifest(
        id = "rag.core",
        name = "Local RAG",
        version = "1.0.0",
        description = "Persistent document ingestion and local retrieval-augmented generation context.",
        permissions = setOf(Permission.READ, Permission.WRITE),
    )

    override fun tools(): List<AgentTool> = listOf(SearchTool(), IngestTool())

    private inner class SearchTool : AgentTool {
        override val definition = ToolDefinition(
            name = "rag_search",
            description = "Search the local knowledge base and return ranked evidence with sources. Input: {query, limit?}",
            category = "knowledge",
            permissions = setOf(Permission.READ),
        )

        override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
            val o = json.parseToJsonElement(argumentsJson).jsonObject
            val query = o["query"]?.jsonPrimitive?.content?.trim().orEmpty()
            require(query.isNotBlank())
            val limit = o["limit"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceIn(1, 16) ?: 8
            val hits = engine.retrieve(query, limit)
            ToolResult(
                ok = hits.isNotEmpty(),
                content = hits.joinToString("\n\n") { hit ->
                    "[${"%.3f".format(hit.score)}] ${hit.chunk.title.ifBlank { hit.chunk.source }}\nSOURCE: ${hit.chunk.source}\n${hit.chunk.text}"
                }.ifBlank { "No matching knowledge found." },
                metadata = mapOf("hits" to hits.size.toString()),
            )
        }.getOrElse { ToolResult(false, "RAG search failed: ${it.message}") }
    }

    private inner class IngestTool : AgentTool {
        override val definition = ToolDefinition(
            name = "rag_ingest",
            description = "Add text to the persistent local knowledge base. Input: {id, source, title?, text}",
            category = "knowledge",
            permissions = setOf(Permission.WRITE),
        )

        override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
            val o = json.parseToJsonElement(argumentsJson).jsonObject
            val id = o["id"]?.jsonPrimitive?.content?.trim().orEmpty()
            val source = o["source"]?.jsonPrimitive?.content?.trim().orEmpty()
            val text = o["text"]?.jsonPrimitive?.content.orEmpty()
            require(id.isNotBlank() && source.isNotBlank() && text.isNotBlank())
            engine.ingest(RagDocument(id, source, o["title"]?.jsonPrimitive?.content.orEmpty(), text))
            ToolResult(true, "Ingested ${text.length} characters from $source")
        }.getOrElse { ToolResult(false, "RAG ingest failed: ${it.message}") }
    }
}
