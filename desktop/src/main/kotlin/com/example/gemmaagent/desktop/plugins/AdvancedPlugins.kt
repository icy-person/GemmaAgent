package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.awt.Rectangle
import java.awt.Robot
import java.awt.Toolkit
import java.awt.event.MouseEvent
import java.awt.image.BufferedImage
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.security.MessageDigest
import java.time.Instant
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO

private val advancedJson = Json { ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }

class SystemPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest("builtin.system", "System & Diagnostics", "1.0.0", "Inspect runtime, storage, environment and processes.", setOf(Permission.READ, Permission.SYSTEM))
    override fun tools(): List<AgentTool> = listOf(SystemInfoTool(workspace), ProcessListTool(), HashTool(workspace))
}

private class SystemInfoTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("system_info", "Return OS, JVM, CPU, memory and workspace information.", "system", setOf(Permission.READ, Permission.SYSTEM))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val r = Runtime.getRuntime()
        val text = buildString {
            appendLine("os=${System.getProperty("os.name")} ${System.getProperty("os.version")}")
            appendLine("arch=${System.getProperty("os.arch")}")
            appendLine("jvm=${System.getProperty("java.version")}")
            appendLine("processors=${r.availableProcessors()}")
            appendLine("memoryMax=${r.maxMemory()}")
            appendLine("memoryFree=${r.freeMemory()}")
            appendLine("memoryTotal=${r.totalMemory()}")
            appendLine("workspace=${workspace.canonicalPath}")
            appendLine("diskFree=${workspace.usableSpace}")
            appendLine("diskTotal=${workspace.totalSpace}")
            appendLine("time=${Instant.now()}")
        }
        ToolResult(true, text)
    }.getOrElse { ToolResult(false, "system_info error: ${it.message}") }
}

private class ProcessListTool : AgentTool {
    override val definition = ToolDefinition("process_list", "List current processes.", "system", setOf(Permission.READ, Permission.SYSTEM))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val processes = ProcessHandle.allProcesses()
            .limit(200)
            .map { handle ->
                val info = handle.info()
                "${handle.pid()} ${info.command().orElse("?")} ${info.commandLine().orElse("").take(160)}"
            }
            .toList()
        ToolResult(true, processes.joinToString("\n").take(50_000), mapOf("count" to processes.size.toString()))
    }.getOrElse { ToolResult(false, "process_list error: ${it.message}") }
}

private class HashTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("hash_file", "Calculate SHA-256 for a workspace file.", "files", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val file = safeFile(workspace, o["path"]?.jsonPrimitive?.content ?: error("path required"))
        require(file.isFile) { "File not found" }
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) { val n = input.read(buf); if (n <= 0) break; digest.update(buf, 0, n) }
        }
        ToolResult(true, digest.digest().joinToString("") { "%02x".format(it) }, mapOf("path" to file.path))
    }.getOrElse { ToolResult(false, "hash error: ${it.message}") }
}

class BrowserPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest("builtin.browser", "Headless Browser", "1.0.0", "Render JavaScript pages and capture screenshots with Chromium.", setOf(Permission.NETWORK, Permission.READ, Permission.WRITE, Permission.EXECUTE))
    override fun tools(): List<AgentTool> = listOf(BrowserRenderTool(), BrowserScreenshotTool(workspace))
}

private object ChromiumLocator {
    fun find(): String? = listOf("chromium", "chromium-browser", "google-chrome", "google-chrome-stable").firstOrNull { name ->
        runCatching { ProcessBuilder("sh", "-lc", "command -v $name").start().inputStream.bufferedReader().use { it.readText().trim().isNotBlank() } }.getOrDefault(false)
    }
}

private class BrowserRenderTool : AgentTool {
    override val definition = ToolDefinition("browser_render", "Render an HTTP(S) page with headless Chromium and execute JavaScript.", "browser", setOf(Permission.NETWORK, Permission.EXECUTE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val url = o["url"]?.jsonPrimitive?.content ?: error("url required")
        require(url.startsWith("https://") || url.startsWith("http://"))
        val chromium = ChromiumLocator.find() ?: error("Chromium not installed")
        val timeout = o["timeout_seconds"]?.jsonPrimitive?.content?.toLongOrNull()?.coerceIn(5, 120) ?: 45L
        val process = ProcessBuilder(chromium, "--headless", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage", "--virtual-time-budget=5000", "--dump-dom", url).redirectErrorStream(true).start()
        val finished = process.waitFor(timeout, TimeUnit.SECONDS)
        if (!finished) process.destroyForcibly()
        val output = process.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(1_500_000) }
        ToolResult(finished && process.exitValueOrMinusOne() == 0, output, mapOf("renderer" to chromium, "url" to url, "javascript" to "true"))
    }.getOrElse { ToolResult(false, "browser_render error: ${it.message}") }
}

private class BrowserScreenshotTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("browser_screenshot", "Capture a rendered page as PNG inside workspace.", "browser", setOf(Permission.NETWORK, Permission.EXECUTE, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val url = o["url"]?.jsonPrimitive?.content ?: error("url required")
        require(url.startsWith("https://") || url.startsWith("http://"))
        val chromium = ChromiumLocator.find() ?: error("Chromium not installed")
        val target = safeFile(workspace, o["output"]?.jsonPrimitive?.content ?: "browser/screenshot.png")
        target.parentFile?.mkdirs()
        val process = ProcessBuilder(chromium, "--headless", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage", "--virtual-time-budget=5000", "--screenshot=${target.path}", "--window-size=1440,900", url).redirectErrorStream(true).start()
        val finished = process.waitFor(60, TimeUnit.SECONDS)
        if (!finished) process.destroyForcibly()
        val log = process.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(20_000) }
        require(finished && target.isFile) { "Screenshot failed: ${log.take(1000)}" }
        ToolResult(true, "screenshot=${target.relativeTo(workspace).path}", mapOf("path" to target.path, "url" to url))
    }.getOrElse { ToolResult(false, "browser_screenshot error: ${it.message}") }
}

class ComputerUsePlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest("builtin.computer", "Computer Use", "1.0.0", "Desktop screenshot, mouse and keyboard control.", setOf(Permission.READ, Permission.WRITE, Permission.SYSTEM))
    override fun tools(): List<AgentTool> = listOf(ScreenCaptureTool(workspace), MouseTool(), KeyboardTool())
}

private class ScreenCaptureTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("screen_capture", "Capture the current desktop screen to PNG.", "computer", setOf(Permission.READ, Permission.WRITE, Permission.SYSTEM))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val target = safeFile(workspace, o["output"]?.jsonPrimitive?.content ?: "computer/screen.png")
        target.parentFile?.mkdirs()
        val size = Toolkit.getDefaultToolkit().screenSize
        val image: BufferedImage = Robot().createScreenCapture(Rectangle(size))
        ImageIO.write(image, "png", target)
        ToolResult(true, "screen=${target.relativeTo(workspace).path}", mapOf("width" to image.width.toString(), "height" to image.height.toString()))
    }.getOrElse { ToolResult(false, "screen_capture error: ${it.message}") }
}

private class MouseTool : AgentTool {
    override val definition = ToolDefinition("mouse", "Move or click the desktop mouse.", "computer", setOf(Permission.SYSTEM), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val robot = Robot()
        val x = o["x"]?.jsonPrimitive?.content?.toIntOrNull() ?: error("x required")
        val y = o["y"]?.jsonPrimitive?.content?.toIntOrNull() ?: error("y required")
        robot.mouseMove(x, y)
        when (o["action"]?.jsonPrimitive?.content ?: "move") {
            "move" -> Unit
            "click" -> { robot.mousePress(MouseEvent.BUTTON1_DOWN_MASK); robot.mouseRelease(MouseEvent.BUTTON1_DOWN_MASK) }
            "right_click" -> { robot.mousePress(MouseEvent.BUTTON3_DOWN_MASK); robot.mouseRelease(MouseEvent.BUTTON3_DOWN_MASK) }
            else -> error("Unknown mouse action")
        }
        ToolResult(true, "mouse action complete")
    }.getOrElse { ToolResult(false, "mouse error: ${it.message}") }
}

private class KeyboardTool : AgentTool {
    override val definition = ToolDefinition("keyboard", "Type text or press a key code.", "computer", setOf(Permission.SYSTEM), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val robot = Robot()
        val text = o["text"]?.jsonPrimitive?.content
        if (text != null) {
            text.forEach { c ->
                when (c) {
                    ' ' -> { robot.keyPress(java.awt.event.KeyEvent.VK_SPACE); robot.keyRelease(java.awt.event.KeyEvent.VK_SPACE) }
                    '\n' -> { robot.keyPress(java.awt.event.KeyEvent.VK_ENTER); robot.keyRelease(java.awt.event.KeyEvent.VK_ENTER) }
                    else -> { val code = java.awt.event.KeyEvent.getExtendedKeyCodeForChar(c.code); require(code != java.awt.event.KeyEvent.VK_UNDEFINED); robot.keyPress(code); robot.keyRelease(code) }
                }
            }
        } else {
            val key = o["keycode"]?.jsonPrimitive?.content?.toIntOrNull() ?: error("text or keycode required")
            robot.keyPress(key); robot.keyRelease(key)
        }
        ToolResult(true, "keyboard action complete")
    }.getOrElse { ToolResult(false, "keyboard error: ${it.message}") }
}

class WorkspaceAnalysisPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest("builtin.analysis", "Workspace Analysis", "1.0.0", "Analyze CSV/TSV data.", setOf(Permission.READ))
    override fun tools(): List<AgentTool> = listOf(CsvAnalyzeTool(workspace))
}

private class CsvAnalyzeTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("analyze_csv", "Summarize a CSV/TSV file.", "analysis", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val file = safeFile(workspace, o["path"]?.jsonPrimitive?.content ?: error("path required"))
        require(file.isFile)
        val delimiter = if (file.extension.lowercase() == "tsv") '\t' else ','
        val lines = file.readLines(StandardCharsets.UTF_8).take(10_001)
        require(lines.isNotEmpty()) { "Empty file" }
        val headers = splitCsv(lines.first(), delimiter)
        val rows = lines.drop(1).filter { it.isNotBlank() }.map { splitCsv(it, delimiter) }
        val missing = IntArray(headers.size)
        val numeric = Array(headers.size) { mutableListOf<Double>() }
        rows.forEach { row -> headers.indices.forEach { i -> val v = row.getOrNull(i)?.trim().orEmpty(); if (v.isBlank()) missing[i]++ else v.toDoubleOrNull()?.let { numeric[i] += it } } }
        val report = buildString { appendLine("rows=${rows.size}"); appendLine("columns=${headers.size}"); headers.forEachIndexed { i, h -> append("$h: missing=${missing[i]}"); if (numeric[i].isNotEmpty()) append(", numeric_count=${numeric[i].size}, min=${numeric[i].minOrNull()}, max=${numeric[i].maxOrNull()}, mean=${numeric[i].average()}"); appendLine() } }
        ToolResult(true, report, mapOf("rows" to rows.size.toString(), "columns" to headers.size.toString()))
    }.getOrElse { ToolResult(false, "analyze_csv error: ${it.message}") }
}

@Serializable
private data class Checkpoint(val id: String, val task: String, val state: String, val createdAt: Long = System.currentTimeMillis())

class CheckpointPlugin(private val file: File) : AgentPlugin {
    override val manifest = PluginManifest("builtin.checkpoints", "Task Checkpoints", "1.0.0", "Persist task checkpoints.", setOf(Permission.READ, Permission.WRITE))
    override fun tools(): List<AgentTool> = listOf(CheckpointTool(file))
}

private class CheckpointTool(private val file: File) : AgentTool {
    override val definition = ToolDefinition("checkpoint", "Save/list/load/delete task checkpoints.", "agent", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "list"
        val states = load()
        when (action) {
            "save" -> { val id = o["id"]?.jsonPrimitive?.content ?: System.currentTimeMillis().toString(); persist(states.filterNot { it.id == id } + Checkpoint(id, o["task"]?.jsonPrimitive?.content.orEmpty(), o["state"]?.jsonPrimitive?.content.orEmpty())); ToolResult(true, "checkpoint=$id") }
            "list" -> ToolResult(true, states.joinToString("\n") { "${it.id}: ${it.task.take(120)}" })
            "load" -> states.firstOrNull { it.id == o["id"]?.jsonPrimitive?.content }?.let { ToolResult(true, advancedJson.encodeToString(Checkpoint.serializer(), it)) } ?: ToolResult(false, "checkpoint not found")
            "delete" -> { val id = o["id"]?.jsonPrimitive?.content ?: error("id required"); persist(states.filterNot { it.id == id }); ToolResult(true, "deleted=$id") }
            else -> ToolResult(false, "Unknown checkpoint action")
        }
    }.getOrElse { ToolResult(false, "checkpoint error: ${it.message}") }

    private fun load(): List<Checkpoint> = runCatching { if (file.isFile) advancedJson.decodeFromString<List<Checkpoint>>(file.readText(StandardCharsets.UTF_8)) else emptyList() }.getOrDefault(emptyList())
    private fun persist(items: List<Checkpoint>) { file.parentFile?.mkdirs(); val tmp = File(file.parentFile ?: error("missing parent"), file.name + ".part"); tmp.writeText(advancedJson.encodeToString(items), StandardCharsets.UTF_8); require(tmp.renameTo(file)) }
}

private fun safeFile(workspace: File, path: String): File {
    val base = workspace.canonicalFile
    val out = File(workspace, path).canonicalFile
    require(out.path == base.path || out.path.startsWith(base.path + File.separator)) { "Path escapes workspace" }
    return out
}

private fun splitCsv(line: String, delimiter: Char): List<String> {
    val out = mutableListOf<String>(); val current = StringBuilder(); var quoted = false; var i = 0
    while (i < line.length) { val c = line[i]; when { c == '"' -> if (quoted && i + 1 < line.length && line[i + 1] == '"') { current.append('"'); i++ } else quoted = !quoted; c == delimiter && !quoted -> { out += current.toString(); current.setLength(0) }; else -> current.append(c) }; i++ }
    out += current.toString(); return out
}

private fun Process.exitValueOrMinusOne(): Int = runCatching { exitValue() }.getOrDefault(-1)

class AdvancedPlugins(private val workspace: File, private val dataDir: File) : AgentPlugin {
    override val manifest = PluginManifest("builtin.advanced-suite", "Advanced Runtime", "1.0.0", "System diagnostics, browser rendering, computer use, data analysis and checkpoints.", setOf(Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.NETWORK, Permission.SYSTEM))
    override fun tools(): List<AgentTool> = buildList {
        addAll(SystemPlugin(workspace).tools())
        addAll(BrowserPlugin(workspace).tools())
        addAll(ComputerUsePlugin(workspace).tools())
        addAll(WorkspaceAnalysisPlugin(workspace).tools())
        addAll(CheckpointPlugin(File(dataDir, "checkpoints.json")).tools())
    }
}
