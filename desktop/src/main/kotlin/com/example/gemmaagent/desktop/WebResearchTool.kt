package com.example.gemmaagent.desktop

import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class WebResearchTool(
    private val engine: WebResearchEngine = WebResearchEngine(),
) : AgentTool {
    private val json = Json { ignoreUnknownKeys = true }

    override val definition = ToolDefinition(
        name = "web_research",
        description = "Research the public web. Searches, downloads pages, extracts main content from HTML, ranks relevant passages, removes duplicates/noise, follows relevant links, and returns source URLs with evidence. Input JSON: {query:string, max_results?:number, crawl_depth?:number}",
        category = "web",
        permissions = setOf(Permission.NETWORK),
        dangerous = false,
    )

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val args = json.parseToJsonElement(argumentsJson).jsonObject
        val query = args["query"]?.jsonPrimitive?.content?.trim().orEmpty()
        require(query.isNotBlank()) { "query is required" }
        val maxResults = args["max_results"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceIn(1, 12) ?: 8
        val crawlDepth = args["crawl_depth"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceIn(0, 1) ?: 1
        val report = engine.research(
            query,
            WebResearchEngine.Config(
                maxResults = maxResults,
                crawlDepth = crawlDepth,
                maxTotalChars = 70_000,
            ),
        )
        val out = buildString {
            appendLine("WEB RESEARCH REPORT")
            appendLine("Query: $query")
            appendLine("Sources: ${report.sources.size}")
            appendLine()
            report.sources.forEachIndexed { index, source ->
                appendLine("[${index + 1}] ${source.title}")
                appendLine("URL: ${source.url}")
                source.chunks.take(8).forEach { chunk ->
                    appendLine("EVIDENCE (${"%.2f".format(chunk.score)}): ${chunk.text}")
                }
                appendLine()
            }
            appendLine("TOP EVIDENCE")
            report.focusedEvidence.take(18).forEachIndexed { index, evidence ->
                appendLine("${index + 1}. $evidence")
            }
        }
        ToolResult(
            ok = report.sources.isNotEmpty(),
            content = out.take(90_000),
            metadata = mapOf(
                "sources" to report.sources.size.toString(),
                "query" to query,
                "crawlDepth" to crawlDepth.toString(),
                "engine" to "local-dom-ranker",
            ),
        )
    }.getOrElse { ToolResult(false, "Web research failed: ${it.message}") }
}
