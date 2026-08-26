package com.example.gemmaagent.shared

import kotlinx.serialization.json.Json
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.util.concurrent.TimeUnit

private val json = Json { ignoreUnknownKeys = true }

actual fun platformTools(workspace: String): List<AgentTool> = listOf(
    JvmFileTool(File(workspace).apply { mkdirs() }),
    JvmHttpTool(),
    JvmProcessTool(),
)

private class JvmFileTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("filesystem", "Read, write, list and delete files inside the configured workspace.", "files", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "list"
        val target = safe(o["path"]?.jsonPrimitive?.content ?: ".")
        when (action) {
            "read" -> ToolResult(true, target.readText().take(100_000))
            "write" -> { target.parentFile?.mkdirs(); target.writeText(o["content"]?.jsonPrimitive?.content.orEmpty()); ToolResult(true, "written=${target.length()}") }
            "append" -> { target.parentFile?.mkdirs(); target.appendText(o["content"]?.jsonPrimitive?.content.orEmpty()); ToolResult(true, "appended") }
            "delete" -> { require(target != root) { "Cannot delete workspace root" }; ToolResult(target.deleteRecursively(), "deleted") }
            "exists" -> ToolResult(true, target.exists().toString())
            "list" -> ToolResult(true, target.listFiles()?.joinToString("\n") { if (it.isDirectory) "[DIR] ${it.name}" else "${it.name} (${it.length()} bytes)" } ?: "empty")
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

private class JvmHttpTool : AgentTool {
    override val definition = ToolDefinition("http", "GET or POST a URL and return bounded text response.", "network", setOf(Permission.NETWORK), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val rawUrl = o["url"]?.jsonPrimitive?.content ?: error("url required")
        val uri = URI(rawUrl)
        require(uri.scheme == "http" || uri.scheme == "https") { "Only HTTP(S) URLs are supported" }
        val method = (o["method"]?.jsonPrimitive?.content ?: "GET").uppercase()
        require(method == "GET" || method == "POST") { "Only GET and POST are supported" }
        val c = uri.toURL().openConnection() as HttpURLConnection
        try {
            c.requestMethod = method
            c.connectTimeout = 10_000
            c.readTimeout = 20_000
            c.instanceFollowRedirects = true
            if (method == "POST") {
                c.doOutput = true
                c.setRequestProperty("Content-Type", "text/plain; charset=utf-8")
                c.outputStream.use { it.write(o["body"]?.jsonPrimitive?.content.orEmpty().toByteArray()) }
            }
            val code = c.responseCode
            val body = (if (code in 200..399) c.inputStream else c.errorStream)
                ?.bufferedReader()?.use { it.readText().take(50_000) }.orEmpty()
            ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("status" to code.toString(), "url" to rawUrl))
        } finally {
            c.disconnect()
        }
    }.getOrElse { ToolResult(false, "HTTP error: ${it.message}") }
}

private class JvmProcessTool : AgentTool {
    override val definition = ToolDefinition("process", "Run a command in the configured workspace and return stdout/stderr.", "development", setOf(Permission.EXECUTE), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val command = o["command"]?.jsonPrimitive?.content ?: error("command required")
        val timeoutSeconds = o["timeout_seconds"]?.jsonPrimitive?.content?.toLongOrNull()?.coerceIn(1, 300) ?: 60L
        val p = ProcessBuilder("sh", "-lc", command).redirectErrorStream(true).start()
        val output = p.inputStream.bufferedReader().use { reader ->
            if (p.waitFor(timeoutSeconds, TimeUnit.SECONDS)) reader.readText().take(50_000)
            else { p.destroyForcibly(); "process timeout after ${timeoutSeconds}s\n" + reader.readText().take(50_000) }
        }
        val code = if (p.isAlive) -1 else p.exitValue()
        ToolResult(code == 0, "exit=$code\n$output", metadata = mapOf("exitCode" to code.toString()))
    }.getOrElse { ToolResult(false, "process error: ${it.message}") }
}
