package com.example.gemmaagent.shared

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import kotlinx.serialization.json.Json
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

private val json = Json { ignoreUnknownKeys = true }

actual fun platformTools(workspace: String): List<AgentTool> = listOf(
    AndroidFileTool(File(workspace).apply { mkdirs() }),
    AndroidHttpTool(),
    AndroidDeviceTool(),
)

private class AndroidFileTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("filesystem", "Read, write, list and delete files inside the configured workspace.", "files", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "list"
        val relative = o["path"]?.jsonPrimitive?.content ?: "."
        val target = safe(relative)
        when (action) {
            "read" -> ToolResult(true, target.readText())
            "write" -> { target.parentFile?.mkdirs(); target.writeText(o["content"]?.jsonPrimitive?.content.orEmpty()); ToolResult(true, "written ${target.length()} bytes") }
            "append" -> { target.parentFile?.mkdirs(); target.appendText(o["content"]?.jsonPrimitive?.content.orEmpty()); ToolResult(true, "appended") }
            "delete" -> { require(target != root) { "Cannot delete workspace root" }; ToolResult(target.deleteRecursively(), "deleted=${target.name}") }
            "exists" -> ToolResult(true, target.exists().toString())
            "list" -> ToolResult(true, target.listFiles()?.joinToString("\n") { if (it.isDirectory) "[DIR] ${it.name}" else "${it.name} (${it.length()} bytes)" } ?: "empty")
            "mkdir" -> { target.mkdirs(); ToolResult(true, "created") }
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

private class AndroidHttpTool : AgentTool {
    override val definition = ToolDefinition("http", "GET or POST a URL and return bounded text response.", "network", setOf(Permission.NETWORK), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val url = o["url"]?.jsonPrimitive?.content ?: error("url required")
        val method = o["method"]?.jsonPrimitive?.content?.uppercase() ?: "GET"
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.requestMethod = method
        connection.connectTimeout = 10_000; connection.readTimeout = 20_000
        if (method == "POST") { connection.doOutput = true; connection.outputStream.use { it.write(o["body"]?.jsonPrimitive?.content.orEmpty().toByteArray()) } }
        val code = connection.responseCode
        val stream = if (code in 200..399) connection.inputStream else connection.errorStream
        val body = stream?.bufferedReader()?.use { it.readText().take(50_000) } ?: ""
        connection.disconnect()
        ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("status" to code.toString()))
    }.getOrElse { ToolResult(false, "HTTP error: ${it.message}") }
}

private class AndroidDeviceTool : AgentTool {
    override val definition = ToolDefinition("device", "Return Android device/runtime information and perform safe URI intents.", "android", setOf(Permission.READ, Permission.SYSTEM))
    override suspend fun execute(argumentsJson: String): ToolResult = ToolResult(true, "Android tool is available. For intent actions the Android UI adapter must supply a Context.")
}

fun copyToClipboard(context: Context, text: String) {
    val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    manager.setPrimaryClip(ClipData.newPlainText("GemmaAgent", text))
}

fun openUri(context: Context, uri: String) {
    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(uri)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}
