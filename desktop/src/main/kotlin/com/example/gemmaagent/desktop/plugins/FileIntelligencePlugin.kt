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
import java.io.FileInputStream
import java.io.BufferedInputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.security.MessageDigest
import java.util.zip.ZipFile
import java.util.zip.GZIPInputStream

class FileIntelligencePlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.file-intelligence",
        name = "File Intelligence",
        version = "1.0.0",
        description = "Inspect file metadata, type signatures, hashes and safely list/extract common archives.",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.EXECUTE),
    )

    override fun tools(): List<AgentTool> = listOf(FileInspectorTool(workspace), ArchiveTool(workspace))
}

private val fileJson = Json { ignoreUnknownKeys = true }

private class FileInspectorTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition(
        name = "inspect_file",
        description = "Inspect a file: size, timestamps, permissions, MIME/type, extension, magic bytes, SHA-256 and basic image dimensions when available. Input: {path:string}",
        category = "files",
        permissions = setOf(Permission.READ),
    )

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val obj = fileJson.parseToJsonElement(argumentsJson).jsonObject
        val file = safe(obj["path"]?.jsonPrimitive?.content ?: error("path required"))
        require(file.isFile) { "Not a regular file: ${file.path}" }
        val bytes = ByteArray(32)
        val read = FileInputStream(file).use { it.read(bytes) }
        val mime = Files.probeContentType(file.toPath()) ?: "unknown"
        val ext = file.extension.ifBlank { "none" }
        val mode = buildString {
            append(if (file.canRead()) "r" else "-")
            append(if (file.canWrite()) "w" else "-")
            append(if (file.canExecute()) "x" else "-")
        }
        val magic = bytes.copyOf(read.coerceAtLeast(0)).joinToString(" ") { "%02x".format(it) }
        val sha = sha256(file)
        val imageInfo = runCatching {
            javax.imageio.ImageIO.read(file)?.let { "${it.width}x${it.height}" }
        }.getOrNull() ?: "n/a"
        ToolResult(true, buildString {
            appendLine("path=${file.canonicalPath}")
            appendLine("name=${file.name}")
            appendLine("extension=$ext")
            appendLine("mime=$mime")
            appendLine("sizeBytes=${file.length()}")
            appendLine("modifiedEpochMs=${file.lastModified()}")
            appendLine("permissions=$mode")
            appendLine("magic=$magic")
            appendLine("sha256=$sha")
            appendLine("imageDimensions=$imageInfo")
        })
    }.getOrElse { ToolResult(false, "inspect_file error: ${it.message}") }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        BufferedInputStream(FileInputStream(file)).use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun safe(path: String): File {
        val root = workspace.canonicalFile
        val out = File(workspace, path).canonicalFile
        require(out.path == root.path || out.path.startsWith(root.path + File.separator)) { "Path escapes workspace" }
        return out
    }
}

private class ArchiveTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition(
        name = "archive",
        description = "Open common archives. Actions: list or extract. Supports ZIP and TAR/GZ/TGZ when tar is installed. Input: {action:'list'|'extract', path:string, output_dir?:string}",
        category = "files",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.EXECUTE),
        dangerous = false,
    )

    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val obj = fileJson.parseToJsonElement(argumentsJson).jsonObject
        val action = obj["action"]?.jsonPrimitive?.content ?: "list"
        val archive = safe(obj["path"]?.jsonPrimitive?.content ?: error("path required"))
        require(archive.isFile) { "Archive not found" }
        when {
            archive.name.endsWith(".zip", true) -> zip(action, archive, obj["output_dir"]?.jsonPrimitive?.content)
            archive.name.endsWith(".tar", true) || archive.name.endsWith(".tar.gz", true) || archive.name.endsWith(".tgz", true) -> tar(action, archive, obj["output_dir"]?.jsonPrimitive?.content)
            else -> ToolResult(false, "Unsupported archive type: ${archive.name}")
        }
    }.getOrElse { ToolResult(false, "archive error: ${it.message}") }

    private fun zip(action: String, archive: File, output: String?): ToolResult {
        ZipFile(archive).use { zip ->
            if (action == "list") {
                val lines = zip.entries().asSequence().take(500).map { e -> if (e.isDirectory) "[DIR] ${e.name}" else "${e.name} (${e.size} bytes)" }.toList()
                return ToolResult(true, lines.joinToString("\n"), metadata = mapOf("entries" to zip.size().toString()))
            }
            require(action == "extract") { "Unknown archive action: $action" }
            val target = safe(output ?: (archive.nameWithoutExtension + "_extracted"))
            target.mkdirs()
            zip.entries().asSequence().forEach { entry ->
                val out = File(target, entry.name).canonicalFile
                require(out.path == target.canonicalPath || out.path.startsWith(target.canonicalPath + File.separator)) { "Archive entry escapes output directory: ${entry.name}" }
                if (entry.isDirectory) out.mkdirs() else {
                    out.parentFile?.mkdirs()
                    zip.getInputStream(entry).use { input -> out.outputStream().use { input.copyTo(it) } }
                }
            }
            return ToolResult(true, "extracted=${target.canonicalPath}")
        }
    }

    private fun tar(action: String, archive: File, output: String?): ToolResult {
        require(action == "list" || action == "extract") { "Unknown archive action: $action" }
        val target = safe(output ?: (archive.name.substringBeforeLast(".tar").substringBeforeLast(".") + "_extracted"))
        val cmd = if (action == "list") {
            listOf("tar", "-tf", archive.canonicalPath)
        } else {
            target.mkdirs()
            listOf("tar", "-xf", archive.canonicalPath, "-C", target.canonicalPath, "--no-same-owner", "--no-same-permissions")
        }
        val p = ProcessBuilder(cmd).redirectErrorStream(true).start()
        val finished = p.waitFor(120, java.util.concurrent.TimeUnit.SECONDS)
        if (!finished) p.destroyForcibly()
        val outputText = p.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(50_000) }
        return ToolResult(finished && p.exitValue() == 0, outputText.ifBlank { "extracted=${target.canonicalPath}" })
    }

    private fun safe(path: String): File {
        val root = workspace.canonicalFile
        val out = File(workspace, path).canonicalFile
        require(out.path == root.path || out.path.startsWith(root.path + File.separator)) { "Path escapes workspace" }
        return out
    }
}
