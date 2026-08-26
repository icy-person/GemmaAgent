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
import java.io.File
import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit

private val docJson = Json { ignoreUnknownKeys = true }

class DocumentAndDataPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.documents-data",
        name = "Documents & Data",
        version = "1.0.0",
        description = "Extract PDF/DOCX/HTML/text, query SQLite through the local CLI and probe media metadata.",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.EXECUTE),
    )
    override fun tools(): List<AgentTool> = listOf(DocumentExtractTool(workspace), SqliteTool(workspace), MediaProbeTool(workspace))
}

private class DocumentExtractTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("extract_document", "Extract text from txt/md/html/pdf/docx using safe local parsers/CLI tools.", "documents", setOf(Permission.READ, Permission.EXECUTE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = docJson.parseToJsonElement(argumentsJson).jsonObject
        val file = safe(o["path"]?.jsonPrimitive?.content ?: error("path required"))
        require(file.isFile) { "File not found" }
        when (file.extension.lowercase()) {
            "txt", "md", "csv", "tsv", "json", "xml", "html", "htm" -> ToolResult(true, file.readText(StandardCharsets.UTF_8).take(200_000))
            "pdf" -> runCli("pdftotext", listOf("-layout", file.path, "-"), 120)
            "docx" -> runCli("sh", listOf("-lc", "unzip -p '${shell(file.path)}' word/document.xml | sed 's/<[^>]*>/ /g'"), 60)
            else -> ToolResult(false, "Unsupported document extension: ${file.extension}")
        }
    }.getOrElse { ToolResult(false, "extract_document error: ${it.message}") }

    private fun runCli(exe: String, args: List<String>, seconds: Long): ToolResult {
        val p = ProcessBuilder(exe, *args.toTypedArray()).redirectErrorStream(true).start()
        val finished = p.waitFor(seconds, TimeUnit.SECONDS)
        if (!finished) p.destroyForcibly()
        val output = p.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(200_000) }
        val code = runCatching { p.exitValue() }.getOrDefault(-1)
        return ToolResult(finished && code == 0, "exit=$code\n$output", metadata = mapOf("exitCode" to code.toString()))
    }

    private fun safe(path: String): File {
        val base = root.canonicalFile
        val out = File(root, path).canonicalFile
        require(out.path == base.path || out.path.startsWith(base.path + File.separator)) { "Path escapes workspace" }
        return out
    }

    private fun shell(value: String) = value.replace("'", "'\\''")
}

private class SqliteTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("sqlite", "Run a bounded read-only SQLite query with sqlite3; write operations require AUTONOMOUS mode approval.", "database", setOf(Permission.READ, Permission.WRITE, Permission.EXECUTE), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = docJson.parseToJsonElement(argumentsJson).jsonObject
        val db = safe(o["path"]?.jsonPrimitive?.content ?: error("path required"))
        val sql = o["sql"]?.jsonPrimitive?.content ?: error("sql required")
        require(sql.length <= 20_000) { "SQL too long" }
        val write = Regex("^\\s*(insert|update|delete|drop|alter|create|replace|vacuum|reindex)", RegexOption.IGNORE_CASE).containsMatchIn(sql)
        val mode = o["mode"]?.jsonPrimitive?.content ?: if (write) "write" else "read"
        if (write) require(mode == "write") { "Explicit mode=write is required for mutating SQL" }
        val temp = File.createTempFile("gemmaagent-sql-", ".txt")
        try {
            val process = ProcessBuilder("sqlite3", "-header", "-column", db.path, sql).redirectErrorStream(true).redirectOutput(temp).start()
            val finished = process.waitFor(90, TimeUnit.SECONDS)
            if (!finished) process.destroyForcibly()
            val code = runCatching { process.exitValue() }.getOrDefault(-1)
            val output = temp.readText(StandardCharsets.UTF_8).take(100_000)
            ToolResult(finished && code == 0, "exit=$code\n$output", metadata = mapOf("mutating" to write.toString()))
        } finally { temp.delete() }
    }.getOrElse { ToolResult(false, "sqlite error: ${it.message}") }

    private fun safe(path: String): File {
        val base = root.canonicalFile
        val out = File(root, path).canonicalFile
        require(out.path == base.path || out.path.startsWith(base.path + File.separator)) { "Path escapes workspace" }
        return out
    }
}

private class MediaProbeTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("media_probe", "Read audio/video/image metadata with ffprobe when installed.", "media", setOf(Permission.READ, Permission.EXECUTE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = docJson.parseToJsonElement(argumentsJson).jsonObject
        val file = safe(o["path"]?.jsonPrimitive?.content ?: error("path required"))
        require(file.isFile)
        val p = ProcessBuilder("ffprobe", "-v", "error", "-show_format", "-show_streams", file.path).redirectErrorStream(true).start()
        val finished = p.waitFor(60, TimeUnit.SECONDS)
        if (!finished) p.destroyForcibly()
        val output = p.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(100_000) }
        val code = runCatching { p.exitValue() }.getOrDefault(-1)
        ToolResult(finished && code == 0, "exit=$code\n$output")
    }.getOrElse { ToolResult(false, "media_probe error: ${it.message}") }

    private fun safe(path: String): File {
        val base = root.canonicalFile
        val out = File(root, path).canonicalFile
        require(out.path == base.path || out.path.startsWith(base.path + File.separator)) { "Path escapes workspace" }
        return out
    }
}
