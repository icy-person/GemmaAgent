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
        description = "Research the public web using a local browser-capable engine. It can execute JavaScript with Chromium, parse the rendered DOM, rank evidence, deduplicate sources and follow relevant links. Input JSON: {query:string, max_results?:number, crawl_depth?:number, javascript?:boolean}",
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
        val javascript = args["javascript"]?.jsonPrimitive?.content?.toBooleanStrictOrNull() ?: true
        val report = engine.research(
            query,
            WebResearchEngine.Config(
                maxResults = maxResults,
                crawlDepth = crawlDepth,
                maxTotalChars = 70_000,
                javascriptEnabled = javascript,
            ),
        )
        val rendered = report.sources.count { it.renderedWithJavaScript }
        val out = buildString {
            appendLine("WEB RESEARCH REPORT")
            appendLine("Query: $query")
            appendLine("Search provider: ${report.searchProvider}")
            appendLine("Sources: ${report.sources.size}")
            appendLine("JavaScript renderer available: ${report.javascriptAvailable}")
            appendLine("Pages rendered with JavaScript: $rendered")
            appendLine()
            report.sources.forEachIndexed { index, source ->
                appendLine("[${index + 1}] ${source.title}")
                appendLine("URL: ${source.url}")
                appendLine("Rendered with JS: ${source.renderedWithJavaScript}")
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
                "javascript" to javascript.toString(),
                "javascriptRenderer" to rendered.toString(),
                "searchProvider" to report.searchProvider,
                "engine" to "chromium-dom-ranker",
            ),
        )
    }.getOrElse { ToolResult(false, "Web research failed: ${it.message}") }
}
