package com.example.gemmaagent.desktop

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.Checkbox
import androidx.compose.material.Divider
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Slider
import androidx.compose.material.Tab
import androidx.compose.material.TabRow
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
import java.awt.EventQueue
import java.io.File
import javax.swing.JFileChooser

private enum class ModelState { IDLE, LOADING, READY, FAILED, CLOSED }

private class DesktopModelRunner(private val path: String) : com.example.gemmaagent.shared.ModelRunner, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    @Volatile private var state = ModelState.IDLE
    fun state(): ModelState = state

    suspend fun start() = withContext(Dispatchers.IO) {
        check(state != ModelState.READY) { "Model is already loaded" }
        state = ModelState.LOADING
        try {
            val modelFile = File(path)
            require(modelFile.isFile) { "Model file does not exist: $path" }
            require(modelFile.canRead()) { "Model file is not readable: $path" }
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
        val activeEngine = engine ?: error("Model is not initialized")
        conversation?.close()
        conversation = activeEngine.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(
                    "You are GemmaAgent, a local autonomous agent. Use available tools, verify tool results, and do not invent evidence."
                )
            )
        )
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        require(prompt.isNotBlank()) { "Prompt must not be blank" }
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

private fun formatLiveEvent(event: AgentEvent): String = when (event) {
    is AgentEvent.Started -> "▶ شروع کار: ${event.task.take(120)}"
    is AgentEvent.Stage -> "• ${event.name}: ${event.detail.take(180)}"
    is AgentEvent.Thinking -> "🧠 در حال پردازش مرحله ${event.iteration}"
    is AgentEvent.ModelOutput -> "◌ خروجی مدل دریافت شد (${event.text.length} chars)"
    is AgentEvent.ToolRequested -> "🛠 اجرای ابزار: ${event.call.name} ${event.call.argumentsJson.take(180)}"
    is AgentEvent.ToolCompleted -> "✓ ابزار ${event.call.name}: ${if (event.result.ok) "موفق" else "خطا"} (${event.result.durationMs} ms) — ${event.result.content.take(220)}"
    is AgentEvent.Reflection -> "↻ بررسی نتیجه: ${event.text.take(220)}"
    is AgentEvent.Finished -> "■ ${if (event.success) "کار تمام شد" else "کار متوقف شد"}"
    is AgentEvent.Failed -> "✕ خطا: ${event.message.take(220)}"
}

fun main() = application {
    Window(onCloseRequest = ::exitApplication, title = "GemmaAgent Desktop") {
        var tab by remember { mutableStateOf(0) }
        var modelPath by remember { mutableStateOf("") }
        var task by remember { mutableStateOf("") }
        var answer by remember { mutableStateOf("") }
        var status by remember { mutableStateOf("No model loaded") }
        var runner by remember { mutableStateOf<DesktopModelRunner?>(null) }
        var agent by remember { mutableStateOf<AgentEngine?>(null) }
        val memory = remember { JvmMemoryStore() }
        val pluginRegistry = remember { PluginRegistry() }
        val metrics = remember { AgentMetrics() }
        val scope = rememberCoroutineScope()
        var mode by remember { mutableStateOf(AgentMode.ASSISTED) }
        var maxIterations by remember { mutableStateOf(30f) }
        var memoryTopK by remember { mutableStateOf(8f) }
        var skillsEnabled by remember { mutableStateOf(true) }
        var reflectionEnabled by remember { mutableStateOf(true) }
        var learnFailures by remember { mutableStateOf(true) }
        var researchEnabled by remember { mutableStateOf(true) }
        var running by remember { mutableStateOf(false) }
        val events = remember { mutableStateOf(listOf<String>()) }
        val observer = remember {
            object : AgentObserver {
                override fun onEvent(event: AgentEvent) {
                    metrics.onEvent(event)
                    val line = formatLiveEvent(event)
                    EventQueue.invokeLater {
                        events.value = (events.value + line).takeLast(160)
                        when (event) {
                            is AgentEvent.Started -> status = "Agent started"
                            is AgentEvent.Thinking -> status = "Thinking — iteration ${event.iteration}"
                            is AgentEvent.ToolRequested -> status = "Running ${event.call.name}"
                            is AgentEvent.ToolCompleted -> status = "Tool ${event.call.name}: ${if (event.result.ok) "done" else "failed"}"
                            is AgentEvent.Reflection -> status = "Verifying result"
                            is AgentEvent.Finished -> status = if (event.success) "Task completed" else "Task stopped"
                            is AgentEvent.Failed -> status = "Task failed"
                            is AgentEvent.Stage -> status = event.name
                            is AgentEvent.ModelOutput -> status = "Model responded"
                        }
                    }
                }
            }
        }

        fun rebuildAgent() {
            val r = runner ?: return
            scope.launch {
                val pluginTools = if (researchEnabled) {
                    runCatching { pluginRegistry.allTools() }.getOrDefault(emptyList())
                } else emptyList()
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
                        skillTopK = if (skillsEnabled) 5 else 0,
                        reflectionEnabled = reflectionEnabled,
                        learnFromFailures = learnFailures,
                        mode = mode,
                    ),
                    observer = observer,
                )
            }
        }

        scope.launch { runCatching { pluginRegistry.install(WebResearchPlugin()) } }

        MaterialTheme {
            Row(Modifier.fillMaxSize()) {
                Column(Modifier.width(210.dp).fillMaxHeight().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("GemmaAgent", style = MaterialTheme.typography.h5)
                    Text("Local autonomous agent")
                    Divider()
                    listOf("Agent", "Research", "Model", "Memory", "Tools", "Learning", "Settings", "Logs").forEachIndexed { index, title ->
                        Button(onClick = { tab = index }, Modifier.fillMaxWidth()) { Text(title) }
                    }
                }
                Column(Modifier.fillMaxSize().padding(16.dp)) {
                    TabRow(selectedTabIndex = tab.coerceIn(0, 7)) {
                        listOf("Agent", "Research", "Model", "Memory", "Tools", "Learning", "Settings", "Logs").forEachIndexed { i, title ->
                            Tab(tab == i, { tab = i }, text = { Text(title) })
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                    when (tab) {
                        0 -> AgentPage(
                            task = task,
                            onTask = { task = it },
                            answer = answer,
                            status = status,
                            enabled = agent != null && !running,
                            running = running,
                            events = events.value,
                        ) {
                            running = true
                            events.value = (events.value + "▶ اجرای Task شروع شد").takeLast(160)
                            scope.launch {
                                runCatching { agent!!.run(task).answer }
                                    .onSuccess { answer = it }
                                    .onFailure { answer = "Error: ${it.message}" }
                                running = false
                            }
                        }
                        1 -> ResearchPage(task, { task = it }, answer, researchEnabled) {
                            tab = 0
                            if (!running) {
                                running = true
                                scope.launch {
                                    runCatching {
                                        agent!!.run(
                                            "Research this topic thoroughly on the public web and produce a cited report: $task"
                                        ).answer
                                    }
                                        .onSuccess { answer = it }
                                        .onFailure { answer = "Research error: ${it.message}" }
                                    running = false
                                }
                            }
                        }
                        2 -> ModelPage(modelPath, { modelPath = it }, status, {
                            val chooser = JFileChooser().apply {
                                dialogTitle = "Select Gemma 4 E4B .litertlm model"
                            }
                            if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                                modelPath = chooser.selectedFile.absolutePath
                                scope.launch {
                                    runCatching {
                                        status = "Loading Gemma 4 E4B (CPU safe mode)..."
                                        val r = DesktopModelRunner(modelPath)
                                        r.start()
                                        runner?.close()
                                        runner = r
                                        rebuildAgent()
                                        status = "Gemma 4 E4B ready (CPU / 8K context / cache disabled)"
                                    }.onFailure {
                                        status = "Load failed: ${it::class.simpleName}: ${it.message}"
                                    }
                                }
                            }
                        }, {
                            runner?.close()
                            runner = null
                            agent = null
                            status = "Model unloaded"
                        })
                        3 -> SimpleInfoPage("Memory", "Persistent experiences, facts and learned workflows are used by the agent across runs.")
                        4 -> SimpleInfoPage("Tools", "Calculator, date/time, filesystem, file search, HTTP, process execution and the Web Research plugin.")
                        5 -> LearningPage(learnFailures, { learnFailures = it; rebuildAgent() }, skillsEnabled, { skillsEnabled = it; rebuildAgent() }, reflectionEnabled, { reflectionEnabled = it; rebuildAgent() })
                        6 -> SettingsPage(mode, { mode = it; rebuildAgent() }, maxIterations, { maxIterations = it; rebuildAgent() }, memoryTopK, { memoryTopK = it; rebuildAgent() }, researchEnabled, { researchEnabled = it; rebuildAgent() })
                        7 -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
                            Text("Metrics: ${metrics.snapshot()}")
                            Spacer(Modifier.height(8.dp))
                            Text(events.value.takeLast(120).joinToString("\n"))
                        }
                    }
                }
            }
        }
    }
}

@Composable private fun AgentPage(
    task: String,
    onTask: (String) -> Unit,
    answer: String,
    status: String,
    enabled: Boolean,
    running: Boolean,
    events: List<String>,
    run: () -> Unit,
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text("Agent", style = MaterialTheme.typography.h5)
            Text(if (running) "● LIVE" else "○ Idle")
        }
        Text(status)
        OutlinedTextField(task, onTask, Modifier.fillMaxWidth(), label = { Text("Task") }, minLines = 5)
        Button(onClick = run, enabled = enabled && task.isNotBlank()) { Text(if (running) "Running…" else "Run Agent") }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Live activity", style = MaterialTheme.typography.h6)
                Text(
                    events.takeLast(40).joinToString("\n").ifBlank { "Waiting for an agent run…" },
                    Modifier.fillMaxWidth().height(220.dp).verticalScroll(rememberScrollState()),
                )
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                Text("Final answer", style = MaterialTheme.typography.h6)
                Text(answer.ifBlank { "No final answer yet." })
            }
        }
    }
}

@Composable private fun ResearchPage(task: String, onTask: (String) -> Unit, answer: String, enabled: Boolean, run: () -> Unit) { Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Web Research", style = MaterialTheme.typography.h5); Text("Search, render JavaScript pages, extract relevant evidence and cite sources."); OutlinedTextField(task, onTask, Modifier.fillMaxWidth(), label = { Text("Research topic") }, minLines = 5); Button(onClick = run, enabled = enabled && task.isNotBlank()) { Text("Research") }; Card(Modifier.fillMaxSize()) { Text(answer, Modifier.padding(14.dp).verticalScroll(rememberScrollState())) } } }
@Composable private fun ModelPage(path: String, onPath: (String) -> Unit, status: String, choose: () -> Unit, unload: () -> Unit) { Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Model", style = MaterialTheme.typography.h5); Text(status); OutlinedTextField(path, onPath, Modifier.fillMaxWidth(), label = { Text(".litertlm model path") }); Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { Button(onClick = choose) { Text("Import / Load") }; OutlinedButton(onClick = unload) { Text("Unload") } } } }
@Composable private fun LearningPage(learnFailures: Boolean, onFailures: (Boolean) -> Unit, skills: Boolean, onSkills: (Boolean) -> Unit, reflection: Boolean, onReflection: (Boolean) -> Unit) { Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Learning", style = MaterialTheme.typography.h5); CheckRow("Learn from failed runs", learnFailures, onFailures); CheckRow("Learn reusable skills", skills, onSkills); CheckRow("Self-reflection / verification", reflection, onReflection); Text("Learning updates external memory and skills; Gemma weights remain unchanged.") } }
@Composable private fun SettingsPage(mode: AgentMode, onMode: (AgentMode) -> Unit, maxIterations: Float, onIterations: (Float) -> Unit, memoryTopK: Float, onMemory: (Float) -> Unit, research: Boolean, onResearch: (Boolean) -> Unit) { Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Settings", style = MaterialTheme.typography.h5); AgentMode.values().forEach { m -> Row(verticalAlignment = Alignment.CenterVertically) { RadioButton(mode == m, { onMode(m) }); Text(m.name) } }; Text("Max iterations: ${maxIterations.toInt()}"); Slider(maxIterations, onIterations, valueRange = 1f..50f); Text("Memory retrieval: ${memoryTopK.toInt()}"); Slider(memoryTopK, onMemory, valueRange = 0f..20f); CheckRow("Web Research plugin enabled", research, onResearch) } }
@Composable private fun CheckRow(label: String, checked: Boolean, onChecked: (Boolean) -> Unit) { Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(checked, onChecked); Text(label) } }
@Composable private fun SimpleInfoPage(title: String, body: String) { Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) { Text(title, style = MaterialTheme.typography.h5); Text(body) } }
