package com.example.gemmaagent.shared

import kotlinx.serialization.json.Json
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit

private val json = Json { ignoreUnknownKeys = true }

actual fun platformTools(workspace: String): List<AgentTool> {
    val root = File(workspace).apply { mkdirs() }
    return listOf(JvmFileTool(root), JvmHttpTool(), JvmProcessTool(root), JvmFindTool(root))
}

private class JvmFileTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("filesystem", "Read/write/append/list/delete/mkdir files inside the workspace", "files", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "list"
        val target = safe(o["path"]?.jsonPrimitive?.content ?: ".")
        when (action) {
            "read" -> ToolResult(true, target.readText(StandardCharsets.UTF_8).take(100_000))
            "write" -> { target.parentFile?.mkdirs(); target.writeText(o["content"]?.jsonPrimitive?.content.orEmpty(), StandardCharsets.UTF_8); ToolResult(true, "written=${target.length()}") }
            "append" -> { target.parentFile?.mkdirs(); target.appendText(o["content"]?.jsonPrimitive?.content.orEmpty(), StandardCharsets.UTF_8); ToolResult(true, "appended=${target.length()}") }
            "delete" -> { require(target != root) { "Cannot delete workspace root" }; ToolResult(target.deleteRecursively(), "deleted") }
            "exists" -> ToolResult(true, target.exists().toString())
            "list" -> {
                val entries = target.listFiles()?.sortedBy { it.name } ?: emptyList()
                ToolResult(true, entries.joinToString("\n") { if (it.isDirectory) "[DIR] ${it.name}" else "${it.name} (${it.length()} bytes)" }.ifBlank { "empty" })
            }
            "mkdir" -> { require(target.mkdirs() || target.isDirectory) { "Cannot create directory" }; ToolResult(true, "created") }
            else -> ToolResult(false, "Unknown filesystem action: $action")
        }
    }.getOrElse { ToolResult(false, "filesystem error: ${it.message}") }

    private fun safe(path: String): File {
        val base = root.canonicalFile
        val out = File(root, path).canonicalFile
        require(out.path == base.path || out.path.startsWith(base.path + File.separator)) { "Path escapes workspace" }
        return out
    }
}

private class JvmFindTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("find_files", "Find files by name or text inside the workspace", "files", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val name = o["name"]?.jsonPrimitive?.content?.trim().orEmpty()
        val contains = o["contains"]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
        val max = o["max_results"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceIn(1, 500) ?: 100
        val hits = mutableListOf<String>()
        root.walkTopDown().forEach { f ->
            if (hits.size >= max || !f.isFile) return@forEach
            if (name.isNotEmpty() && !f.name.contains(name, true)) return@forEach
            if (contains != null) {
                val text = runCatching { f.readText().take(250_000) }.getOrDefault("")
                if (!text.contains(contains, true)) return@forEach
            }
            hits += f.relativeTo(root).path
        }
        ToolResult(true, hits.joinToString("\n"), metadata = mapOf("count" to hits.size.toString()))
    }.getOrElse { ToolResult(false, "find error: ${it.message}") }
}

private class JvmHttpTool : AgentTool {
    override val definition = ToolDefinition("http", "GET or POST a URL with bounded response", "network", setOf(Permission.NETWORK), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val rawUrl = o["url"]?.jsonPrimitive?.content ?: error("url required")
        val uri = URI(rawUrl)
        require(uri.scheme == "http" || uri.scheme == "https") { "Only HTTP(S) URLs are supported" }
        val method = (o["method"]?.jsonPrimitive?.content ?: "GET").uppercase()
        require(method == "GET" || method == "POST") { "Only GET and POST are supported" }
        val c = uri.toURL().openConnection() as HttpURLConnection
        try {
            c.requestMethod = method; c.connectTimeout = 10_000; c.readTimeout = 20_000; c.instanceFollowRedirects = true
            c.setRequestProperty("User-Agent", "GemmaAgent/1.0")
            if (method == "POST") { c.doOutput = true; c.setRequestProperty("Content-Type", "application/json; charset=utf-8"); c.outputStream.use { it.write(o["body"]?.jsonPrimitive?.content.orEmpty().toByteArray(StandardCharsets.UTF_8)) } }
            val code = c.responseCode
            val body = (if (code in 200..399) c.inputStream else c.errorStream)?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText().take(50_000) }.orEmpty()
            ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("status" to code.toString(), "url" to rawUrl))
        } finally { c.disconnect() }
    }.getOrElse { ToolResult(false, "HTTP error: ${it.message}") }
}

private class JvmProcessTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("process", "Run a command in workspace with timeout and bounded output", "development", setOf(Permission.EXECUTE), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val command = o["command"]?.jsonPrimitive?.content ?: error("command required")
        val timeoutSeconds = o["timeout_seconds"]?.jsonPrimitive?.content?.toLongOrNull()?.coerceIn(1, 300) ?: 60L
        require(command.length <= 8_000) { "command too long" }
        val temp = File.createTempFile("gemma-agent-process-", ".log")
        try {
            val process = ProcessBuilder("sh", "-lc", command).directory(root).redirectErrorStream(true).redirectOutput(temp).start()
            val finished = process.waitFor(timeoutSeconds, TimeUnit.SECONDS)
            if (!finished) process.destroyForcibly()
            val code = if (process.isAlive) -1 else process.exitValue()
            val output = temp.readText(StandardCharsets.UTF_8).take(50_000)
            ToolResult(code == 0 && finished, "exit=$code\n$output", metadata = mapOf("exitCode" to code.toString(), "timedOut" to (!finished).toString()))
        } finally { runCatching { temp.delete() } }
    }.getOrElse { ToolResult(false, "process error: ${it.message}") }
}
