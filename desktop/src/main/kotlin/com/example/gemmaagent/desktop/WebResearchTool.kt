package com.example.gemmaagent.desktop

import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.regex.Pattern

class WebResearchTool(
    private val maxResults: Int = 8,
    private val maxPageChars: Int = 12_000,
) : AgentTool {
    override val definition = ToolDefinition(
        name = "web_research",
        description = "Search the public web and collect readable text from the top results. Input JSON: {query:string, max_results?:number}",
        category = "web",
        permissions = setOf(Permission.NETWORK),
        dangerous = false,
    )

    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(15))
        .followRedirects(HttpClient.Redirect.NORMAL)
        .build()

    override suspend fun execute(argumentsJson: String): ToolResult {
        return runCatching {
            val args = Json.parseToJsonElement(argumentsJson).jsonObject
            val query = args["query"]?.jsonPrimitive?.content?.trim().orEmpty()
            require(query.isNotBlank()) { "query is required" }
            val requested = args["max_results"]?.jsonPrimitive?.content?.toIntOrNull() ?: maxResults
            val limit = requested.coerceIn(1, maxResults)
            val links = search(query).take(limit)
            val gathered = links.mapNotNull { result ->
                runCatching { fetchPage(result.first)?.let { result.second to it } }.getOrNull()
            }
            val out = buildString {
                appendLine("WEB RESEARCH: $query")
                gathered.forEachIndexed { index, item ->
                    appendLine("\n[${index + 1}] ${item.first}")
                    appendLine(item.second)
                }
                if (gathered.isEmpty()) appendLine("No readable pages were retrieved.")
            }
            ToolResult(true, out.take(maxPageChars * limit), metadata = mapOf("sources" to gathered.size.toString()))
        }.getOrElse { ToolResult(false, "Web research failed: ${it.message}") }
    }

    private fun search(query: String): List<Pair<String, String>> {
        val url = "https://html.duckduckgo.com/html/?q=" + java.net.URLEncoder.encode(query, Charsets.UTF_8)
        val html = get(url)
        val pattern = Pattern.compile("<a[^>]+class=\\\"result__a\\\"[^>]+href=\\\"([^\\\"]+)\\\"[^>]*>(.*?)</a>", Pattern.CASE_INSENSITIVE)
        val matcher = pattern.matcher(html)
        val results = mutableListOf<Pair<String, String>>()
        while (matcher.find() && results.size < maxResults) {
            val rawUrl = matcher.group(1)
            val title = clean(matcher.group(2))
            val resolved = if (rawUrl.startsWith("//")) "https:$rawUrl" else rawUrl
            if (resolved.startsWith("http")) results += resolved to title
        }
        return results.distinctBy { it.first }
    }

    private fun fetchPage(url: String): String {
        val html = get(url)
        val noScript = html.replace(Regex("(?is)<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>"), " ")
        val text = noScript
            .replace(Regex("(?is)<br\\s*/?>"), "\n")
            .replace(Regex("(?is)</p>|</div>|</article>|</li>|</h[1-6]>"), "\n")
            .replace(Regex("(?is)<[^>]+>"), " ")
            .replace(Regex("&nbsp;"), " ")
            .replace(Regex("&amp;"), "&")
            .replace(Regex("&quot;"), "\"")
            .replace(Regex("&#39;"), "'")
            .replace(Regex("\\s+"), " ")
            .trim()
        return text.take(maxPageChars)
    }

    private fun get(url: String): String {
        val request = HttpRequest.newBuilder(URI(url))
            .timeout(Duration.ofSeconds(20))
            .header("User-Agent", "GemmaAgent/1.0 research client")
            .header("Accept", "text/html,application/xhtml+xml")
            .GET()
            .build()
        return client.send(request, HttpResponse.BodyHandlers.ofString()).body()
    }

    private fun clean(html: String): String = html.replace(Regex("<[^>]+>"), " ").replace(Regex("\\s+"), " ").trim()
}
