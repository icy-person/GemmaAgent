package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import java.net.HttpURLConnection
import java.net.URI
import java.nio.charset.StandardCharsets

class GitHubPlugin : AgentPlugin {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    override val manifest = PluginManifest(
        id = "github.public",
        name = "GitHub & CI",
        version = "2.0.0",
        description = "Read public GitHub repositories and, when GITHUB_TOKEN is configured and approval allows it, create issues/PRs and dispatch Actions workflows.",
        permissions = setOf(Permission.NETWORK, Permission.READ, Permission.WRITE),
    )

    override fun tools(): List<AgentTool> = listOf(GitHubTool())

    private inner class GitHubTool : AgentTool {
        override val definition = ToolDefinition(
            name = "github",
            description = "GitHub access. Actions: repo, search_repo, issues, file, create_issue, create_pr, dispatch_workflow. Write actions require GITHUB_TOKEN.",
            category = "development",
            permissions = setOf(Permission.NETWORK, Permission.READ, Permission.WRITE),
            dangerous = true,
        )

        override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
            val o = json.parseToJsonElement(argumentsJson).jsonObject
            when (o["action"]?.jsonPrimitive?.content ?: "search_repo") {
                "repo" -> get("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}")
                "search_repo" -> get("https://api.github.com/search/repositories?q=${enc(required(o, "query"))}&per_page=${limit(o)}")
                "issues" -> get("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}/issues?per_page=${limit(o)}")
                "file" -> get("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}/contents/${required(o, "path")}")
                "create_issue" -> post("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}/issues", buildJsonObject {
                    put("title", required(o, "title"))
                    put("body", o["body"]?.jsonPrimitive?.content.orEmpty())
                })
                "create_pr" -> post("https://api.github.com/repos/${required(o, "owner")}/${required(o, "repo")}/pulls", buildJsonObject {
                    put("title", required(o, "title"))
                    put("head", required(o, "head"))
                    put("base", required(o, "base"))
                    put("body", o["body"]?.jsonPrimitive?.content.orEmpty())
                    put("draft", o["draft"]?.jsonPrimitive?.content?.toBoolean() ?: false)
                })
                "dispatch_workflow" -> {
                    val owner = required(o, "owner"); val repo = required(o, "repo"); val workflow = required(o, "workflow"); val ref = o["ref"]?.jsonPrimitive?.content ?: "main"
                    post("https://api.github.com/repos/$owner/$repo/actions/workflows/$workflow/dispatches", buildJsonObject { put("ref", ref) })
                }
                else -> error("Unknown GitHub action")
            }
        }.getOrElse { ToolResult(false, "GitHub error: ${it.message}") }

        private fun required(o: kotlinx.serialization.json.JsonObject, key: String): String = o[key]?.jsonPrimitive?.content?.trim()?.takeIf { it.isNotBlank() }
            ?: error("$key is required")

        private fun limit(o: kotlinx.serialization.json.JsonObject): Int = o["limit"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceIn(1, 20) ?: 10
        private fun enc(value: String): String = java.net.URLEncoder.encode(value, StandardCharsets.UTF_8)

        private fun connection(rawUrl: String, write: Boolean = false): HttpURLConnection {
            val uri = URI(rawUrl)
            require(uri.scheme == "https" && uri.host == "api.github.com") { "Only api.github.com is allowed" }
            return (uri.toURL().openConnection() as HttpURLConnection).apply {
                connectTimeout = 10_000
                readTimeout = 30_000
                requestMethod = if (write) "POST" else "GET"
                setRequestProperty("Accept", "application/vnd.github+json")
                setRequestProperty("X-GitHub-Api-Version", "2022-11-28")
                setRequestProperty("User-Agent", "GemmaAgent/1.0")
                if (write) {
                    val token = System.getenv("GITHUB_TOKEN")?.trim().orEmpty()
                    require(token.isNotBlank()) { "GITHUB_TOKEN is required for write actions" }
                    setRequestProperty("Authorization", "Bearer $token")
                    setRequestProperty("Content-Type", "application/json; charset=utf-8")
                    doOutput = true
                }
            }
        }

        private fun get(rawUrl: String): ToolResult {
            val c = connection(rawUrl)
            return try {
                val code = c.responseCode
                val body = (if (code in 200..399) c.inputStream else c.errorStream)?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText().take(80_000) }.orEmpty()
                ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("status" to code.toString(), "url" to rawUrl))
            } finally { c.disconnect() }
        }

        private fun post(rawUrl: String, body: kotlinx.serialization.json.JsonObject): ToolResult {
            val c = connection(rawUrl, write = true)
            return try {
                c.outputStream.use { it.write(body.toString().toByteArray(StandardCharsets.UTF_8)) }
                val code = c.responseCode
                val response = (if (code in 200..399) c.inputStream else c.errorStream)?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText().take(80_000) }.orEmpty()
                ToolResult(code in 200..299, "HTTP $code\n$response", metadata = mapOf("status" to code.toString(), "url" to rawUrl))
            } finally { c.disconnect() }
        }
    }
}
