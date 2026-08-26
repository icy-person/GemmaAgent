package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.HttpURLConnection
import java.net.URI
import java.nio.charset.StandardCharsets

class GitHubPlugin : AgentPlugin {
    private val json = Json { ignoreUnknownKeys = true }
    override val manifest = PluginManifest(
        id = "github.public",
        name = "GitHub Public",
        version = "1.0.0",
        description = "Search and inspect public GitHub repositories, issues and source files.",
        permissions = setOf(Permission.NETWORK, Permission.READ),
    )

    override fun tools(): List<AgentTool> = listOf(GitHubTool())

    private inner class GitHubTool : AgentTool {
        override val definition = ToolDefinition(
            name = "github",
            description = "Public GitHub access. Input {action:'repo'|'search_repo'|'issues'|'file', owner?, repo?, query?, path?, limit?}.",
            category = "development",
            permissions = setOf(Permission.NETWORK, Permission.READ),
        )

        override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
            val o = json.parseToJsonElement(argumentsJson).jsonObject
            when (o["action"]?.jsonPrimitive?.content ?: "search_repo") {
                "repo" -> get("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}")
                "search_repo" -> get("https://api.github.com/search/repositories?q=${enc(required(o, "query"))}&per_page=${limit(o)}")
                "issues" -> get("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}/issues?per_page=${limit(o)}")
                "file" -> get("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}/contents/${required(o, "path")}")
                else -> error("Unknown GitHub action")
            }
        }.getOrElse { ToolResult(false, "GitHub error: ${it.message}") }

        private fun required(o: kotlinx.serialization.json.JsonObject, key: String): String = o[key]?.jsonPrimitive?.content?.trim()?.takeIf { it.isNotBlank() }
            ?: error("$key is required")

        private fun limit(o: kotlinx.serialization.json.JsonObject): Int = o["limit"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceIn(1, 20) ?: 10
        private fun enc(value: String): String = java.net.URLEncoder.encode(value, StandardCharsets.UTF_8)

        private fun get(rawUrl: String): ToolResult {
            val uri = URI(rawUrl)
            require(uri.scheme == "https" && uri.host == "api.github.com") { "Only api.github.com is allowed" }
            val c = uri.toURL().openConnection() as HttpURLConnection
            return try {
                c.connectTimeout = 10_000
                c.readTimeout = 20_000
                c.setRequestProperty("Accept", "application/vnd.github+json")
                c.setRequestProperty("User-Agent", "GemmaAgent/1.0")
                val code = c.responseCode
                val body = (if (code in 200..399) c.inputStream else c.errorStream)?.bufferedReader()?.use { it.readText().take(80_000) }.orEmpty()
                ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("status" to code.toString(), "url" to rawUrl))
            } finally { c.disconnect() }
        }
    }
}
