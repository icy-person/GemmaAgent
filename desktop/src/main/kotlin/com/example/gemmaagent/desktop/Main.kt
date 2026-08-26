package com.example.gemmaagent.desktop

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.Divider
import androidx.compose.material.IconButton
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Slider
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.material.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import com.example.gemmaagent.desktop.plugins.WebResearchPlugin
import com.example.gemmaagent.shared.AgentConfig
import com.example.gemmaagent.shared.AgentEngine
import com.example.gemmaagent.shared.AgentEvent
import com.example.gemmaagent.shared.AgentMetrics
import com.example.gemmaagent.shared.AgentMode
import com.example.gemmaagent.shared.AgentObserver
import com.example.gemmaagent.shared.CalculatorTool
import com.example.gemmaagent.shared.DateTimeTool
import com.example.gemmaagent.shared.EchoTool
import com.example.gemmaagent.shared.JvmMemoryStore
import com.example.gemmaagent.shared.PluginRegistry
import com.example.gemmaagent.shared.platformTools
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import javax.swing.JFileChooser

private enum class ModelState { IDLE, LOADING, READY, FAILED, CLOSED }
private data class ChatMessage(val role: String, val text: String)

private class DesktopModelRunner(private val path: String) : com.example.gemmaagent.shared.ModelRunner, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    @Volatile private var state = ModelState.IDLE

    suspend fun start() = withContext(Dispatchers.IO) {
        check(state != ModelState.READY) { "Model is already loaded" }
        state = ModelState.LOADING
        try {
            val modelFile = File(path)
            require(modelFile.isFile && modelFile.canRead()) { "Model file is not readable: $path" }
            require(modelFile.length() > 0L) { "Model file is empty: $path" }
            require(path.endsWith(".litertlm", ignoreCase = true)) { "Expected a .litertlm model" }
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(threadCount = 2),
                visionBackend = null,
                audioBackend = null,
                maxNumTokens = 8192,
                cacheDir = ":nocache",
            )
            val newEngine = Engine(config)
            newEngine.initialize()
            engine = newEngine
            reset()
            state = ModelState.READY
        } catch (t: Throwable) {
            runCatching { conversation?.close() }
            runCatching { engine?.close() }
            conversation = null
            engine = null
            state = ModelState.FAILED
            throw t
        }
    }

    override suspend fun reset() = withContext(Dispatchers.IO) {
        val active = engine ?: error("Model is not initialized")
        conversation?.close()
        conversation = active.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(
                    "You are GemmaAgent, a local autonomous agent. Use available tools, verify results, and do not invent evidence."
                )
            )
        )
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        check(state == ModelState.READY) { "Model is not ready" }
        conversation?.sendMessage(prompt)?.toString() ?: error("Conversation is not initialized")
    }

    override fun close() {
        runCatching { conversation?.close() }
        runCatching { engine?.close() }
        conversation = null
        engine = null
        state = ModelState.CLOSED
    }
}

private fun eventText(event: AgentEvent): String = when (event) {
    is AgentEvent.Started -> "Started · ${event.task.take(120)}"
    is AgentEvent.Stage -> "${event.name} · ${event.detail.take(180)}"
    is AgentEvent.Thinking -> "Thinking · step ${event.iteration}"
    is AgentEvent.ModelOutput -> "Model output · ${event.text.take(180)}"
    is AgentEvent.ToolRequested -> "Tool · ${event.call.name} ${event.call.argumentsJson.take(180)}"
    is AgentEvent.ToolCompleted -> "Tool result · ${event.call.name} · ${if (event.result.ok) "success" else "failed"} · ${event.result.durationMs} ms"
    is AgentEvent.Reflection -> "Verification · ${event.text.take(180)}"
    is AgentEvent.Finished -> "Finished · ${if (event.success) "success" else "stopped"}"
    is AgentEvent.Failed -> "Error · ${event.message.take(180)}"
}

fun main() = application {
    Window(onCloseRequest = ::exitApplication, title = "GemmaAgent") {
        var dark by remember { mutableStateOf(true) }
        var activeChat by remember { mutableStateOf("New chat") }
        var task by remember { mutableStateOf("") }
        var modelPath by remember { mutableStateOf("") }
        var runner by remember { mutableStateOf<DesktopModelRunner?>(null) }
        var agent by remember { mutableStateOf<AgentEngine?>(null) }
        var status by remember { mutableStateOf("Load Gemma 4 E4B to begin") }
        var running by remember { mutableStateOf(false) }
        var showActivity by remember { mutableStateOf(true) }
        var mode by remember { mutableStateOf(AgentMode.ASSISTED) }
        var maxIterations by remember { mutableStateOf(30f) }
        var memoryTopK by remember { mutableStateOf(8f) }

        val messages = remember { mutableStateListOf<ChatMessage>() }
        val activity = remember { mutableStateListOf<String>() }
        val memory = remember { JvmMemoryStore() }
        val registry = remember { PluginRegistry() }
        val metrics = remember { AgentMetrics() }
        val scope = rememberCoroutineScope()

        val observer = remember {
            object : AgentObserver {
                override fun onEvent(event: AgentEvent) {
                    metrics.onEvent(event)
                    activity += eventText(event)
                    while (activity.size > 180) activity.removeAt(0)
                    status = when (event) {
                        is AgentEvent.Started -> "Working…"
                        is AgentEvent.Thinking -> "Thinking · step ${event.iteration}"
                        is AgentEvent.ToolRequested -> "Using ${event.call.name}…"
                        is AgentEvent.ToolCompleted -> "${event.call.name}: ${if (event.result.ok) "done" else "failed"}"
                        is AgentEvent.Reflection -> "Verifying…"
                        is AgentEvent.Finished -> if (event.success) "Ready" else "Stopped"
                        is AgentEvent.Failed -> "Error"
                        is AgentEvent.Stage -> event.name
                        is AgentEvent.ModelOutput -> "Generating…"
                    }
                }
            }
        }

        fun rebuildAgent() {
            val r = runner ?: return
            scope.launch {
                val pluginTools = runCatching { registry.allTools() }.getOrDefault(emptyList())
                val tools = buildList {
                    add(CalculatorTool())
                    add(DateTimeTool())
                    add(EchoTool())
                    addAll(platformTools(File(modelPath).parent ?: "."))
                    addAll(pluginTools)
                }
                agent = AgentEngine(
                    model = r,
                    memory = memory,
                    tools = tools,
                    config = AgentConfig(
                        maxIterations = maxIterations.toInt(),
                        memoryTopK = memoryTopK.toInt(),
                        skillTopK = 5,
                        reflectionEnabled = true,
                        learnFromFailures = true,
                        mode = mode,
                    ),
                    observer = observer,
                )
            }
        }

        scope.launch { runCatching { registry.install(WebResearchPlugin()) } }

        MaterialTheme {
            Surface(Modifier.fillMaxSize(), color = if (dark) Color(0xFF212121) else Color(0xFFFFFFFF)) {
                Row(Modifier.fillMaxSize()) {
                    ChatSidebar(activeChat, { activeChat = it }, {
                        activeChat = "New chat"
                        messages.clear()
                        activity.clear()
                        task = ""
                    }, dark) { dark = !dark }

                    Column(Modifier.weight(1f).fillMaxHeight()) {
                        ChatHeader(status, runner != null, { showActivity = !showActivity }, {
                            val chooser = JFileChooser().apply { dialogTitle = "Select Gemma 4 E4B .litertlm model" }
                            if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                                modelPath = chooser.selectedFile.absolutePath
                                scope.launch {
                                    runCatching {
                                        status = "Loading Gemma 4 E4B…"
                                        val r = DesktopModelRunner(modelPath)
                                        r.start()
                                        runner?.close()
                                        runner = r
                                        rebuildAgent()
                                    }.onSuccess { status = "Gemma 4 E4B · CPU · 8K" }
                                        .onFailure { status = "Load failed: ${it::class.simpleName}: ${it.message}" }
                                }
                            }
                        })

                        Row(Modifier.weight(1f).fillMaxWidth()) {
                            Column(Modifier.weight(1f).fillMaxHeight()) {
                                ChatTimeline(messages)
                                Composer(task, { task = it }, agent != null, running) {
                                    val prompt = task.trim()
                                    if (prompt.isEmpty() || agent == null || running) return@Composer
                                    task = ""
                                    messages += ChatMessage("user", prompt)
                                    running = true
                                    showActivity = true
                                    scope.launch {
                                        runCatching { agent!!.run(prompt).answer }
                                            .onSuccess { messages += ChatMessage("assistant", it) }
                                            .onFailure { messages += ChatMessage("assistant", "Error: ${it.message}") }
                                        running = false
                                    }
                                }
                            }
                            if (showActivity) ActivityPanel(activity, metrics.snapshot())
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatSidebar(active: String, onSelect: (String) -> Unit, onNew: () -> Unit, dark: Boolean, toggleTheme: () -> Unit) {
    val bg = if (dark) Color(0xFF171717) else Color(0xFFF7F7F8)
    Column(Modifier.width(260.dp).fillMaxHeight().background(bg).padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = onNew, Modifier.fillMaxWidth()) { Text("+  New chat") }
        Text("Chats", style = MaterialTheme.typography.caption, modifier = Modifier.padding(8.dp))
        TextButton(onClick = { onSelect("GemmaAgent") }, Modifier.fillMaxWidth()) { Text(if (active == "GemmaAgent") "●  GemmaAgent" else "  GemmaAgent") }
        TextButton(onClick = { onSelect("Research") }, Modifier.fillMaxWidth()) { Text(if (active == "Research") "●  Research" else "  Research") }
        Spacer(Modifier.weight(1f))
        Divider()
        TextButton(onClick = toggleTheme, Modifier.fillMaxWidth()) { Text(if (dark) "Light mode" else "Dark mode") }
        Text("GemmaAgent · local", style = MaterialTheme.typography.caption, modifier = Modifier.padding(8.dp))
    }
}

@Composable
private fun ChatHeader(status: String, loaded: Boolean, onActivity: () -> Unit, onModel: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("GemmaAgent", fontWeight = FontWeight.SemiBold)
            Text(status, style = MaterialTheme.typography.caption)
        }
        Text(if (loaded) "● Local" else "○ No model", style = MaterialTheme.typography.caption)
        Spacer(Modifier.width(8.dp))
        TextButton(onClick = onActivity) { Text("Activity") }
        TextButton(onClick = onModel) { Text("Model") }
    }
    Divider()
}

@Composable
private fun ChatTimeline(messages: List<ChatMessage>) {
    Box(Modifier.weight(1f).fillMaxWidth()) {
        if (messages.isEmpty()) {
            Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                Text("What can I help you with?", style = MaterialTheme.typography.h5)
                Spacer(Modifier.height(10.dp))
                Text("Research the web, inspect files, use RAG, work with GitHub, or build code.")
            }
        } else {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp, vertical = 22.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
                messages.forEach { MessageRow(it) }
            }
        }
    }
}

@Composable
private fun MessageRow(message: ChatMessage) {
    val isUser = message.role == "user"
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start) {
        Card(
            modifier = Modifier.widthIn(max = 820.dp),
            shape = androidx.compose.foundation.shape.RoundedCornerShape(18.dp),
            backgroundColor = if (isUser) Color(0xFF303030) else Color.Transparent,
            elevation = 0.dp,
        ) {
            Text(message.text, Modifier.padding(horizontal = 10.dp, vertical = 6.dp))
        }
    }
}

@Composable
private fun Composer(value: String, onValue: (String) -> Unit, ready: Boolean, running: Boolean, send: () -> Unit) {
    Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 24.dp, vertical = 12.dp)) {
        Card(shape = androidx.compose.foundation.shape.RoundedCornerShape(22.dp), elevation = 5.dp) {
            Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.Bottom) {
                OutlinedTextField(
                    value = value,
                    onValueChange = onValue,
                    modifier = Modifier.weight(1f),
                    placeholder = { Text(if (ready) "Message GemmaAgent…" else "Load Gemma 4 E4B first…") },
                    maxLines = 7,
                )
                Spacer(Modifier.width(8.dp))
                IconButton(onClick = send, enabled = ready && !running && value.isNotBlank(), modifier = Modifier.size(48.dp)) {
                    Text(if (running) "…" else "↑", fontWeight = FontWeight.Bold)
                }
            }
        }
        Text("GemmaAgent can make mistakes. Check important results.", style = MaterialTheme.typography.caption, modifier = Modifier.align(Alignment.CenterHorizontally).padding(top = 6.dp))
    }
}

@Composable
private fun ActivityPanel(events: List<String>, metrics: Map<String, Long>) {
    Card(Modifier.width(340.dp).fillMaxHeight().padding(10.dp), shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp), elevation = 2.dp) {
        Column(Modifier.fillMaxSize().padding(14.dp)) {
            Text("Agent activity", style = MaterialTheme.typography.h6, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            Text("Live", style = MaterialTheme.typography.caption)
            Spacer(Modifier.height(8.dp))
            Text(events.takeLast(80).joinToString("\n").ifBlank { "Waiting…" }, Modifier.weight(1f).verticalScroll(rememberScrollState()), style = MaterialTheme.typography.caption)
            Divider(Modifier.padding(vertical = 8.dp))
            Text("thinking=${metrics["thinking"] ?: 0}  tools=${metrics["toolCalls"] ?: 0}", style = MaterialTheme.typography.caption)
            Text("failures=${metrics["toolFailures"] ?: 0}  done=${metrics["completed"] ?: 0}", style = MaterialTheme.typography.caption)
        }
    }
}
