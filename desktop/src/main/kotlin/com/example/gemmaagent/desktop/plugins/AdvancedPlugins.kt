package com.example.gemmaagent.desktop.plugins

import com.example.gemmaagent.shared.AgentPlugin
import com.example.gemmaagent.shared.AgentTool
import com.example.gemmaagent.shared.Permission
import com.example.gemmaagent.shared.PluginManifest
import com.example.gemmaagent.shared.ToolDefinition
import com.example.gemmaagent.shared.ToolResult
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.awt.Color
import java.awt.Rectangle
import java.awt.Robot
import java.awt.Toolkit
import java.awt.image.BufferedImage
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.time.Instant
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO

private val advancedJson = Json { ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }

class SystemPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.system",
        name = "System & Diagnostics",
        version = "1.0.0",
        description = "Inspect runtime, storage, environment and running processes.",
        permissions = setOf(Permission.READ, Permission.SYSTEM),
    )
    override fun tools(): List<AgentTool> = listOf(SystemInfoTool(workspace), ProcessListTool(), HashTool())
}

private class SystemInfoTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("system_info", "Return OS, JVM, CPU, memory, workspace and environment information.", "system", setOf(Permission.READ, Permission.SYSTEM))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val runtime = Runtime.getRuntime()
        val workspacePath = workspace.canonicalPath
        val free = workspace.usableSpace
        val total = workspace.totalSpace
        ToolResult(true, buildString {
            appendLine("os=${System.getProperty("os.name")} ${System.getProperty("os.version")}")
            appendLine("arch=${System.getProperty("os.arch")}")
            appendLine("jvm=${System.getProperty("java.version")}")
            appendLine("processors=${runtime.availableProcessors()}")
            appendLine("memoryMax=${runtime.maxMemory()}")
            appendLine("memoryFree=${runtime.freeMemory()}")
            appendLine("memoryTotal=${runtime.totalMemory()}")
            appendLine("workspace=$workspacePath")
            appendLine("diskFree=$free")
            appendLine("diskTotal=$total")
            appendLine("time=${Instant.now()}")
        })
    }.getOrElse { ToolResult(false, "system_info error: ${it.message}") }
}

private class ProcessListTool : AgentTool {
    override val definition = ToolDefinition("process_list", "List current processes and basic CPU/memory metadata when available.", "system", setOf(Permission.READ, Permission.SYSTEM))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val processes = ProcessHandle.allProcesses().take(200).mapNotNull { handle ->
            runCatching {
                val info = handle.info()
                "${handle.pid()} ${info.command().orElse("?")} ${info.commandLine().orElse("").take(160)}"
            }.getOrNull()
        }
        ToolResult(true, processes.joinToString("\n").take(50_000), metadata = mapOf("count" to processes.size.toString()))
    }.getOrElse { ToolResult(false, "process_list error: ${it.message}") }
}

private class HashTool : AgentTool {
    override val definition = ToolDefinition("hash_file", "Calculate SHA-256 for a file inside the workspace.", "files", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val file = File(o["path"]?.jsonPrimitive?.content ?: error("path required")).canonicalFile
        require(file.isFile) { "File not found" }
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        ToolResult(true, md.digest().joinToString("") { "%02x".format(it) }, metadata = mapOf("path" to file.path))
    }.getOrElse { ToolResult(false, "hash error: ${it.message}") }
}

class BrowserPlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.browser",
        name = "Headless Browser",
        version = "1.0.0",
        description = "Render JavaScript pages, dump final DOM and capture screenshots using installed Chromium.",
        permissions = setOf(Permission.NETWORK, Permission.READ, Permission.WRITE, Permission.EXECUTE),
    )
    override fun tools(): List<AgentTool> = listOf(BrowserRenderTool(workspace), BrowserScreenshotTool(workspace))
}

private object ChromiumLocator {
    fun find(): String? = listOf("chromium", "chromium-browser", "google-chrome", "google-chrome-stable").firstOrNull { name ->
        runCatching { ProcessBuilder("sh", "-lc", "command -v $name").start().inputStream.bufferedReader().use { it.readText().trim().isNotBlank() } }.getOrDefault(false)
    }
}

private class BrowserRenderTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("browser_render", "Open a URL in headless Chromium, execute JavaScript and return the rendered DOM.", "browser", setOf(Permission.NETWORK, Permission.EXECUTE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val url = o["url"]?.jsonPrimitive?.content ?: error("url required")
        require(url.startsWith("https://") || url.startsWith("http://")) { "Only HTTP(S) URLs supported" }
        val chromium = ChromiumLocator.find() ?: error("Chromium not installed")
        val timeout = o["timeout_seconds"]?.jsonPrimitive?.content?.toLongOrNull()?.coerceIn(5, 120) ?: 45L
        val process = ProcessBuilder(
            chromium, "--headless", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
            "--virtual-time-budget=5000", "--dump-dom", url,
        ).redirectErrorStream(true).start()
        val finished = process.waitFor(timeout, TimeUnit.SECONDS)
        if (!finished) process.destroyForcibly()
        val output = process.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(1_500_000) }
        ToolResult(finished && process.exitValueOrMinusOne() == 0, output, metadata = mapOf("renderer" to chromium, "url" to url, "javascript" to "true"))
    }.getOrElse { ToolResult(false, "browser_render error: ${it.message}") }
}

private class BrowserScreenshotTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("browser_screenshot", "Capture a rendered web page to a PNG file inside workspace.", "browser", setOf(Permission.NETWORK, Permission.EXECUTE, Permission.WRITE), dangerous = false)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val url = o["url"]?.jsonPrimitive?.content ?: error("url required")
        require(url.startsWith("https://") || url.startsWith("http://")) { "Only HTTP(S) URLs supported" }
        val chromium = ChromiumLocator.find() ?: error("Chromium not installed")
        val target = File(workspace, o["output"]?.jsonPrimitive?.content ?: "browser/screenshot.png").canonicalFile
        require(target.path.startsWith(workspace.canonicalPath + File.separator)) { "Output escapes workspace" }
        target.parentFile?.mkdirs()
        val process = ProcessBuilder(
            chromium, "--headless", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
            "--virtual-time-budget=5000", "--screenshot=${target.path}", "--window-size=1440,900", url,
        ).redirectErrorStream(true).start()
        val finished = process.waitFor(60, TimeUnit.SECONDS)
        if (!finished) process.destroyForcibly()
        val log = process.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText().take(20_000) }
        require(finished && target.isFile) { "Screenshot failed: ${log.take(1000)}" }
        ToolResult(true, "screenshot=${target.relativeTo(workspace).path}", metadata = mapOf("path" to target.path, "url" to url))
    }.getOrElse { ToolResult(false, "browser_screenshot error: ${it.message}") }
}

class ComputerUsePlugin(private val workspace: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.computer",
        name = "Computer Use",
        version = "1.0.0",
        description = "Desktop screenshot, mouse and keyboard control through java.awt.Robot.",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.SYSTEM),
    )
    override fun tools(): List<AgentTool> = listOf(ScreenCaptureTool(workspace), MouseTool(), KeyboardTool())
}

private class ScreenCaptureTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("screen_capture", "Capture the current desktop screen to PNG.", "computer", setOf(Permission.READ, Permission.WRITE, Permission.SYSTEM), dangerous = false)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val target = File(workspace, o["output"]?.jsonPrimitive?.content ?: "computer/screen.png").canonicalFile
        require(target.path.startsWith(workspace.canonicalPath + File.separator))
        target.parentFile?.mkdirs()
        val size = Toolkit.getDefaultToolkit().screenSize
        val image: BufferedImage = Robot().createScreenCapture(Rectangle(size))
        ImageIO.write(image, "png", target)
        ToolResult(true, "screen=${target.relativeTo(workspace).path}", metadata = mapOf("width" to image.width.toString(), "height" to image.height.toString()))
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
            "click" -> { robot.mousePress(Robot.BUTTON1_DOWN_MASK); robot.mouseRelease(Robot.BUTTON1_DOWN_MASK) }
            "right_click" -> { robot.mousePress(Robot.BUTTON3_DOWN_MASK); robot.mouseRelease(Robot.BUTTON3_DOWN_MASK) }
            else -> error("Unknown mouse action")
        }
        ToolResult(true, "mouse action complete", metadata = mapOf("x" to x.toString(), "y" to y.toString()))
    }.getOrElse { ToolResult(false, "mouse error: ${it.message}") }
}

private class KeyboardTool : AgentTool {
    override val definition = ToolDefinition("keyboard", "Type a string or press a key on the desktop keyboard.", "computer", setOf(Permission.SYSTEM), dangerous = true)
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val robot = Robot()
        val text = o["text"]?.jsonPrimitive?.content
        if (text != null) {
            text.forEach { char ->
                when (char) {
                    ' ' -> { robot.keyPress(java.awt.event.KeyEvent.VK_SPACE); robot.keyRelease(java.awt.event.KeyEvent.VK_SPACE) }
                    '\n' -> { robot.keyPress(java.awt.event.KeyEvent.VK_ENTER); robot.keyRelease(java.awt.event.KeyEvent.VK_ENTER) }
                    else -> {
                        val code = java.awt.event.KeyEvent.getExtendedKeyCodeForChar(char.code)
                        require(code != java.awt.event.KeyEvent.VK_UNDEFINED) { "Unsupported character: $char" }
                        robot.keyPress(code); robot.keyRelease(code)
                    }
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
    override val manifest = PluginManifest(
        id = "builtin.analysis",
        name = "Workspace Analysis",
        version = "1.0.0",
        description = "Analyze CSV/TSV data and produce bounded summaries without external services.",
        permissions = setOf(Permission.READ),
    )
    override fun tools(): List<AgentTool> = listOf(CsvAnalyzeTool(workspace))
}

private class CsvAnalyzeTool(private val workspace: File) : AgentTool {
    override val definition = ToolDefinition("analyze_csv", "Summarize a CSV/TSV file: rows, columns, missing values and numeric min/max/mean.", "analysis", setOf(Permission.READ))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val file = File(workspace, o["path"]?.jsonPrimitive?.content ?: error("path required")).canonicalFile
        require(file.path.startsWith(workspace.canonicalPath + File.separator) && file.isFile)
        val delimiter = if (file.extension.lowercase() == "tsv") '\t' else ','
        val lines = file.readLines(StandardCharsets.UTF_8).take(10_001)
        require(lines.isNotEmpty()) { "Empty file" }
        val headers = splitCsv(lines.first(), delimiter)
        val rows = lines.drop(1).filter { it.isNotBlank() }.map { splitCsv(it, delimiter) }
        val missing = IntArray(headers.size)
        val numeric = Array(headers.size) { mutableListOf<Double>() }
        rows.forEach { row ->
            headers.indices.forEach { i ->
                val value = row.getOrNull(i)?.trim().orEmpty()
                if (value.isBlank()) missing[i]++ else value.toDoubleOrNull()?.let { numeric[i] += it }
            }
        }
        val report = buildString {
            appendLine("rows=${rows.size}")
            appendLine("columns=${headers.size}")
            headers.forEachIndexed { i, h ->
                append("$h: missing=${missing[i]}")
                if (numeric[i].isNotEmpty()) append(", numeric_count=${numeric[i].size}, min=${numeric[i].minOrNull()}, max=${numeric[i].maxOrNull()}, mean=${numeric[i].average()}")
                appendLine()
            }
        }
        ToolResult(true, report, metadata = mapOf("rows" to rows.size.toString(), "columns" to headers.size.toString()))
    }.getOrElse { ToolResult(false, "analyze_csv error: ${it.message}") }
}

private fun splitCsv(line: String, delimiter: Char): List<String> {
    val out = mutableListOf<String>(); val current = StringBuilder(); var quoted = false; var i = 0
    while (i < line.length) {
        val c = line[i]
        when {
            c == '"' -> if (quoted && i + 1 < line.length && line[i + 1] == '"') { current.append('"'); i++ } else quoted = !quoted
            c == delimiter && !quoted -> { out += current.toString(); current.setLength(0) }
            else -> current.append(c)
        }
        i++
    }
    out += current.toString(); return out
}

class CheckpointPlugin(private val file: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.checkpoints",
        name = "Task Checkpoints",
        version = "1.0.0",
        description = "Persist task checkpoints and resume metadata across application restarts.",
        permissions = setOf(Permission.READ, Permission.WRITE),
    )
    override fun tools(): List<AgentTool> = listOf(CheckpointTool(file))
}

@Serializable
private data class Checkpoint(val id: String, val task: String, val state: String, val createdAt: Long = System.currentTimeMillis())

private class CheckpointTool(private val file: File) : AgentTool {
    override val definition = ToolDefinition("checkpoint", "Save/list/load/delete task checkpoints. Actions: save, list, load, delete.", "agent", setOf(Permission.READ, Permission.WRITE))
    override suspend fun execute(argumentsJson: String): ToolResult = runCatching {
        val o = advancedJson.parseToJsonElement(argumentsJson).jsonObject
        val action = o["action"]?.jsonPrimitive?.content ?: "list"
        val states = load()
        when (action) {
            "save" -> {
                val id = o["id"]?.jsonPrimitive?.content ?: System.currentTimeMillis().toString()
                val next = states.filterNot { it.id == id } + Checkpoint(id, o["task"]?.jsonPrimitive?.content ?: "", o["state"]?.jsonPrimitive?.content ?: "")
                persist(next); ToolResult(true, "checkpoint=$id")
            }
            "list" -> ToolResult(true, states.joinToString("\n") { "${it.id}: ${it.task.take(120)}" })
            "load" -> states.firstOrNull { it.id == (o["id"]?.jsonPrimitive?.content ?: "") }?.let { ToolResult(true, advancedJson.encodeToString(Checkpoint.serializer(), it)) } ?: ToolResult(false, "checkpoint not found")
            "delete" -> { val id = o["id"]?.jsonPrimitive?.content ?: error("id required"); persist(states.filterNot { it.id == id }); ToolResult(true, "deleted=$id") }
            else -> ToolResult(false, "Unknown checkpoint action")
        }
    }.getOrElse { ToolResult(false, "checkpoint error: ${it.message}") }

    private fun load(): List<Checkpoint> = runCatching { if (file.isFile) advancedJson.decodeFromString<List<Checkpoint>>(file.readText()) else emptyList() }.getOrDefault(emptyList())
    private fun persist(items: List<Checkpoint>) { file.parentFile?.mkdirs(); val tmp = File(file.parentFile ?: file.absoluteFile.parentFile!!, file.name + ".part"); tmp.writeText(advancedJson.encodeToString(items)); require(tmp.renameTo(file)) }
}

private fun Process.exitValueOrMinusOne(): Int = runCatching { exitValue() }.getOrDefault(-1)

class AdvancedPlugins(private val workspace: File, private val dataDir: File) : AgentPlugin {
    override val manifest = PluginManifest(
        id = "builtin.advanced-suite",
        name = "Advanced Runtime",
        version = "1.0.0",
        description = "System diagnostics, browser rendering, computer use, data analysis and persistent checkpoints.",
        permissions = setOf(Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.NETWORK, Permission.SYSTEM),
    )
    override fun tools(): List<AgentTool> = buildList {
        addAll(SystemPlugin(workspace).tools())
        addAll(BrowserPlugin(workspace).tools())
        addAll(ComputerUsePlugin(workspace).tools())
        addAll(WorkspaceAnalysisPlugin(workspace).tools())
        addAll(CheckpointPlugin(File(dataDir, "checkpoints.json")).tools())
    }
}
