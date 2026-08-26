package com.example.gemmaagent.shared

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import kotlinx.serialization.json.Json
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

private val json = Json { ignoreUnknownKeys = true }

object AndroidAgentContext {
    @Volatile var context: Context? = null
}

actual fun platformTools(workspace: String): List<AgentTool> = listOf(
    AndroidFileTool(File(workspace).apply { mkdirs() }), AndroidHttpTool(), AndroidDeviceTool(),
    AndroidClipboardTool(), AndroidIntentTool(), AndroidPackageTool()
)

private class AndroidFileTool(private val root: File) : AgentTool {
    override val definition = ToolDefinition("filesystem", "Read/write/append/list/delete/mkdir files inside the workspace", "files", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "list"
        val target = safe(o["path"]?.jsonPrimitive?.content ?: ".")
        when (action) {
            "read" -> ToolResult(true, target.readText().take(100_000))
            "write" -> { target.parentFile?.mkdirs(); target.writeText(o["content"]?.jsonPrimitive?.content.orEmpty()); ToolResult(true, "written=${target.length()}") }
            "append" -> { target.parentFile?.mkdirs(); target.appendText(o["content"]?.jsonPrimitive?.content.orEmpty()); ToolResult(true, "appended=${target.length()}") }
            "delete" -> { require(target != root) { "Cannot delete workspace root" }; ToolResult(target.deleteRecursively(), "deleted=${target.name}") }
            "exists" -> ToolResult(true, target.exists().toString())
            "list" -> ToolResult(true, target.listFiles()?.sortedBy { it.name }?.joinToString("\n") { if (it.isDirectory) "[DIR] ${it.name}" else "${it.name} (${it.length()} bytes)" } ?: "empty")
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

private class AndroidHttpTool : AgentTool {
    override val definition = ToolDefinition("http", "GET or POST a URL and return bounded text response", "network", setOf(Permission.NETWORK), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val rawUrl = o["url"]?.jsonPrimitive?.content ?: error("url required")
        val uri = Uri.parse(rawUrl)
        require(uri.scheme == "http" || uri.scheme == "https") { "Only HTTP(S) URLs are allowed" }
        val method = (o["method"]?.jsonPrimitive?.content ?: "GET").uppercase()
        require(method == "GET" || method == "POST") { "Only GET and POST are supported" }
        val c = URL(rawUrl).openConnection() as HttpURLConnection
        try {
            c.requestMethod = method; c.connectTimeout = 10_000; c.readTimeout = 20_000; c.instanceFollowRedirects = true
            c.setRequestProperty("User-Agent", "GemmaAgent/1.0")
            if (method == "POST") { c.doOutput = true; c.setRequestProperty("Content-Type", "application/json; charset=utf-8"); c.outputStream.use { it.write(o["body"]?.jsonPrimitive?.content.orEmpty().toByteArray()) } }
            val code = c.responseCode
            val body = (if (code in 200..399) c.inputStream else c.errorStream)?.bufferedReader()?.use { it.readText().take(50_000) }.orEmpty()
            ToolResult(code in 200..399, "HTTP $code\n$body", metadata = mapOf("status" to code.toString(), "url" to rawUrl))
        } finally { c.disconnect() }
    }.getOrElse { ToolResult(false, "HTTP error: ${it.message}") }
}

private class AndroidDeviceTool : AgentTool {
    override val definition = ToolDefinition("device", "Return Android device/runtime information", "android", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = ToolResult(true, "brand=${Build.BRAND}\nmodel=${Build.MODEL}\nmanufacturer=${Build.MANUFACTURER}\nsdk=${Build.VERSION.SDK_INT}\nabis=${Build.SUPPORTED_ABIS.joinToString()}")
}

private class AndroidClipboardTool : AgentTool {
    override val definition = ToolDefinition("clipboard", "Read or write Android clipboard", "android", setOf(Permission.CLIPBOARD, Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = json.parseToJsonElement(argumentsJson).jsonObject
        val context = AndroidAgentContext.context ?: error("Android context unavailable")
        val manager = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        when (o["action"]?.jsonPrimitive?.content ?: "read") {
            "read" -> ToolResult(true, manager.primaryClip?.getItemAt(0)?.coerceToText(context)?.toString().orEmpty())
            "write" -> { manager.setPrimaryClip(ClipData.newPlainText("GemmaAgent", o["text"]?.jsonPrimitive?.content.orEmpty())); ToolResult(true, "clipboard updated") }
            else -> ToolResult(false, "Unknown clipboard action")
        }
    }.getOrElse { ToolResult(false, "clipboard error: ${it.message}") }
}

private class AndroidIntentTool : AgentTool {
    override val definition = ToolDefinition("intent", "Open an HTTP/HTTPS URL in the user's browser", "android", setOf(Permission.SYSTEM), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val context = AndroidAgentContext.context ?: error("Android context unavailable")
        val value = json.parseToJsonElement(argumentsJson).jsonObject["url"]?.jsonPrimitive?.content ?: error("url required")
        val uri = Uri.parse(value)
        require(uri.scheme == "http" || uri.scheme == "https") { "Only HTTP(S) URLs can be opened" }
        context.startActivity(Intent(Intent.ACTION_VIEW, uri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        ToolResult(true, "intent sent", metadata = mapOf("url" to value))
    }.getOrElse { ToolResult(false, "intent error: ${it.message}") }
}

private class AndroidPackageTool : AgentTool {
    override val definition = ToolDefinition("packages", "List installed Android packages visible to the app", "android", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val context = AndroidAgentContext.context ?: error("Android context unavailable")
        ToolResult(true, context.packageManager.getInstalledApplications(0).map { it.packageName }.sorted().joinToString("\n").take(50_000))
    }.getOrElse { ToolResult(false, "package query error: ${it.message}") }
}
