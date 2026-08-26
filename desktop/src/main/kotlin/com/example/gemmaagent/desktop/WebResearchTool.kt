package com.example.gemmaagent.desktop

import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URI
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern

private val json = Json { ignoreUnknownKeys = true }

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

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val args = json.parseToJsonElement(argumentsJson).jsonObject
        val query = args["query"]?.jsonPrimitive?.content?.trim().orEmpty()
        require(query.isNotBlank()) { "query is required" }
        val requested = args["max_results"]?.jsonPrimitive?.content?.toIntOrNull() ?: maxResults
        val limit = requested.coerceIn(1, maxResults)
        val links = search(query).take(limit)
        val gathered = links.mapNotNull { result ->
            runCatching { Page(result.first, result.second, fetchPage(result.first)) }.getOrNull()
        }
        val out = buildString {
            appendLine("WEB RESEARCH: $query")
            gathered.forEachIndexed { index, item ->
                appendLine("\n[${index + 1}] ${item.title}")
                appendLine("URL: ${item.url}")
                appendLine(item.text)
            }
            if (gathered.isEmpty()) appendLine("No readable pages were retrieved.")
        }
        ToolResult(true, out.take(maxPageChars * limit), metadata = mapOf(
            "sources" to gathered.size.toString(),
            "query" to query,
        ))
    }.getOrElse { ToolResult(false, "Web research failed: ${it.message}") }

    private data class Page(val url: String, val title: String, val text: String)

    private fun search(query: String): List<Pair<String, String>> {
        val url = "https://html.duckduckgo.com/html/?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8)
        val html = get(url)
        val pattern = Pattern.compile("<a[^>]+class=\\\"result__a\\\"[^>]+href=\\\"([^\\\"]+)\\\"[^>]*>(.*?)</a>", Pattern.CASE_INSENSITIVE)
        val matcher = pattern.matcher(html)
        val results = mutableListOf<Pair<String, String>>()
        while (matcher.find() && results.size < maxResults) {
            val rawUrl = matcher.group(1)
            val title = clean(matcher.group(2))
            val resolved = if (rawUrl.startsWith("//")) "https:$rawUrl" else rawUrl
            runCatching {
                val uri = URI(resolved)
                require(uri.scheme == "http" || uri.scheme == "https")
                results += resolved to title
            }
        }
        return results.distinctBy { it.first }
    }

    private fun fetchPage(url: String): String {
        val html = get(url)
        val noScript = html.replace(Regex("(?is)<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>"), " ")
        return noScript
            .replace(Regex("(?is)<br\\s*/?>"), "\n")
            .replace(Regex("(?is)</p>|</div>|</article>|</li>|</h[1-6]>"), "\n")
            .replace(Regex("(?is)<[^>]+>"), " ")
            .replace(Regex("&nbsp;"), " ")
            .replace(Regex("&amp;"), "&")
            .replace(Regex("&quot;"), "\"")
            .replace(Regex("&#39;"), "'")
            .replace(Regex("\\s+"), " ")
            .trim()
            .take(maxPageChars)
    }

    private fun get(url: String): String {
        val uri = URI(url)
        require(uri.scheme == "http" || uri.scheme == "https") { "Only HTTP(S) URLs are supported" }
        val connection = (uri.toURL().openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 15_000
            readTimeout = 20_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "GemmaAgent/1.0 research client")
            setRequestProperty("Accept", "text/html,application/xhtml+xml")
        }
        return try {
            val code = connection.responseCode
            require(code in 200..399) { "HTTP $code" }
            val stream = connection.inputStream
            BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8)).use { it.readText() }
        } finally {
            connection.disconnect()
        }
    }

    private fun clean(html: String): String = html.replace(Regex("<[^>]+>"), " ").replace(Regex("\\s+"), " ").trim()
}
