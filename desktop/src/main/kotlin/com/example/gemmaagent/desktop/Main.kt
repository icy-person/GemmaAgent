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
import androidx.compose.runtime.DisposableEffect
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

private class DesktopModelRunner(private val path: String) : com.example.gemmaagent.shared.ModelRunner, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    @Volatile private var state = ModelState.IDLE

    fun state(): ModelState = state

    suspend fun start() = withContext(Dispatchers.IO) {
        check(state != ModelState.READY) { "Model is already loaded" }
        state = ModelState.LOADING
        try {
            require(File(path).isFile) { "Model file does not exist: $path" }
            require(path.endsWith(".litertlm", ignoreCase = true)) { "Expected a .litertlm model" }
            val newEngine = Engine(EngineConfig(modelPath = path, backend = Backend.CPU()))
            newEngine.initialize()
            engine = newEngine
            reset()
            state = ModelState.READY
        } catch (t: Throwable) {
            runCatching { engine?.close() }
            engine = null
            conversation = null
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
                    """
                    You are GemmaAgent, a local autonomous agent.
                    Use tools when useful. Never claim a tool succeeded without its result.
                    Prefer concise, verifiable actions. Maintain continuity across the current session.
                    """.trimIndent()
                )
            )
        )
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        check(state == ModelState.READY) { "Model is not ready" }
        conversation?.sendMessage(prompt)?.toString() ?: error("Conversation is not initialized")
    }

    suspend fun benchmark(prompt: String = "Reply with exactly: BENCHMARK_OK") : ModelBenchmark = withContext(Dispatchers.Default) {
        check(state == ModelState.READY) { "Model is not ready" }
        val started = System.nanoTime()
        val result = conversation?.sendMessage(prompt)?.toString() ?: error("Conversation is not initialized")
        val totalMs = (System.nanoTime() - started) / 1_000_000L
        val estimatedTokens = result.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.size.coerceAtLeast(1)
        ModelBenchmark(
            firstTokenMs = totalMs,
            totalMs = totalMs,
            estimatedTokens = estimatedTokens,
            tokensPerSecond = estimatedTokens * 1000.0 / totalMs.coerceAtLeast(1),
        )
    }

    override fun close() {
        runCatching { conversation?.close() }
        conversation = null
        runCatching { engine?.close() }
        engine = null
        state = ModelState.CLOSED
    }
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
        var benchmark by remember { mutableStateOf<ModelBenchmark?>(null) }
        var memoryCount by remember { mutableStateOf(0L) }
        val memory = remember { JvmMemoryStore() }
        val modelLibrary = remember { ModelLibrary() }
        val metrics = remember { AgentMetrics() }
        val scope = rememberCoroutineScope()
        var mode by remember { mutableStateOf(AgentMode.ASSISTED) }
        var maxIterations by remember { mutableStateOf(30f) }
        var memoryTopK by remember { mutableStateOf(8f) }
        var skillsEnabled by remember { mutableStateOf(true) }
        var reflectionEnabled by remember { mutableStateOf(true) }
        var learnFailures by remember { mutableStateOf(true) }
        var researchEnabled by remember { mutableStateOf(true) }
        var events by remember { mutableStateOf(listOf<String>()) }

        fun observer(): AgentObserver = object : AgentObserver {
            override fun onEvent(event: AgentEvent) {
                metrics.onEvent(event)
                events = (events + event.toString()).takeLast(150)
            }
        }

        fun rebuildAgent() {
            val currentRunner = runner ?: return
            val tools = buildList {
                add(CalculatorTool())
                add(DateTimeTool())
                add(EchoTool())
                addAll(platformTools(File(modelPath).parent ?: "."))
                if (researchEnabled) add(WebResearchTool())
            }
            agent = AgentEngine(
                model = currentRunner,
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
                observer = observer(),
            )
        }

        DisposableEffect(Unit) {
            modelPath = modelLibrary.lastPath()
            onDispose { runner?.close() }
        }

        MaterialTheme {
            Row(Modifier.fillMaxSize()) {
                Column(Modifier.width(215.dp).fillMaxHeight().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("GemmaAgent", style = MaterialTheme.typography.h5)
                    Text("Gallery-inspired local AI agent")
                    Divider()
                    listOf("Agent", "Model", "Benchmark", "Memory", "Tools", "Learning", "Settings", "Logs").forEachIndexed { index, title ->
                        Button(onClick = { tab = index }, modifier = Modifier.fillMaxWidth()) { Text(title) }
                    }
                    Spacer(Modifier.height(8.dp))
                    Text("Memory: $memoryCount")
                    Text("Tools: ${agent?.let { "loaded" } ?: "none"}")
                }

                Column(Modifier.fillMaxSize().padding(16.dp)) {
                    TabRow(selectedTabIndex = tab.coerceIn(0, 7)) {
                        listOf("Agent", "Model", "Benchmark", "Memory", "Tools", "Learning", "Settings", "Logs").forEachIndexed { i, title ->
                            Tab(selected = tab == i, onClick = { tab = i }, text = { Text(title) })
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                    when (tab) {
                        0 -> AgentPage(task, { task = it }, answer, status, agent != null, { runTask ->
                            scope.launch {
                                runCatching { agent!!.run(runTask) }
                                    .onSuccess { run -> answer = run.answer; status = if (run.success) "Completed • ${run.iterations} iterations" else "Stopped after ${run.iterations} iterations" }
                                    .onFailure { answer = "Error: ${it.message}"; status = "Task failed" }
                            }
                        })
                        1 -> ModelPage(modelPath, { modelPath = it }, status, modelLibrary.sizeBytes(modelPath), runner?.state() ?: ModelState.IDLE, {
                            val chooser = JFileChooser().apply { dialogTitle = "Select Gemma .litertlm model" }
                            if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                                modelPath = chooser.selectedFile.absolutePath
                                scope.launch {
                                    val error = modelLibrary.validate(modelPath)
                                    if (error != null) { status = error; return@launch }
                                    runCatching {
                                        status = "Loading model..."
                                        runner?.close()
                                        val next = DesktopModelRunner(modelPath)
                                        next.start()
                                        runner = next
                                        modelLibrary.remember(modelPath)
                                        rebuildAgent()
                                        status = "Model ready"
                                        benchmark = null
                                    }.onFailure { status = "Load failed: ${it.message}" }
                                }
                            }
                        }, {
                            runner?.close(); runner = null; agent = null; benchmark = null; status = "Model unloaded"
                        })
                        2 -> BenchmarkPage(benchmark, runner != null, {
                            scope.launch {
                                runCatching { runner!!.benchmark() }
                                    .onSuccess { benchmark = it; status = "Benchmark complete" }
                                    .onFailure { status = "Benchmark failed: ${it.message}" }
                            }
                        })
                        3 -> MemoryPage(memoryCount, { scope.launch { memoryCount = memory.count() } })
                        4 -> ToolsPage()
                        5 -> LearningPage(learnFailures, { learnFailures = it; rebuildAgent() }, skillsEnabled, { skillsEnabled = it; rebuildAgent() }, reflectionEnabled, { reflectionEnabled = it; rebuildAgent() })
                        6 -> SettingsPage(mode, { mode = it; rebuildAgent() }, maxIterations, { maxIterations = it; rebuildAgent() }, memoryTopK, { memoryTopK = it; rebuildAgent() }, researchEnabled, { researchEnabled = it; rebuildAgent() })
                        else -> Text(events.takeLast(100).joinToString("\n"), Modifier.fillMaxSize().verticalScroll(rememberScrollState()))
                    }
                }
            }
        }
    }
}

@Composable private fun AgentPage(task: String, onTask: (String) -> Unit, answer: String, status: String, enabled: Boolean, run: (String) -> Unit) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Agent", style = MaterialTheme.typography.h5)
        Text(status)
        OutlinedTextField(task, onTask, Modifier.fillMaxWidth(), label = { Text("Task") }, minLines = 5)
        Button(onClick = { run(task) }, enabled = enabled && task.isNotBlank()) { Text("Run Agent") }
        Card(Modifier.fillMaxWidth().weight(1f)) { Text(answer, Modifier.padding(14.dp).verticalScroll(rememberScrollState())) }
    }
}

@Composable private fun ModelPage(path: String, onPath: (String) -> Unit, status: String, sizeBytes: Long, state: ModelState, choose: () -> Unit, unload: () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Model", style = MaterialTheme.typography.h5)
        Text("State: $state")
        Text(status)
        OutlinedTextField(path, onPath, Modifier.fillMaxWidth(), label = { Text(".litertlm model path") })
        if (sizeBytes > 0) Text("Size: %.2f GB".format(sizeBytes / 1_000_000_000.0))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = choose) { Text("Import / Load") }
            OutlinedButton(onClick = unload, enabled = state != ModelState.IDLE && state != ModelState.CLOSED) { Text("Unload") }
        }
        Text("The model stays external to the application. The app remembers the last imported model path, similar to Gallery's model management flow.")
    }
}

@Composable private fun BenchmarkPage(result: ModelBenchmark?, enabled: Boolean, run: () -> Unit) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Benchmark", style = MaterialTheme.typography.h5)
        Button(onClick = run, enabled = enabled) { Text("Run CPU Benchmark") }
        result?.let {
            Text("First-token estimate: ${it.firstTokenMs} ms")
            Text("Total: ${it.totalMs} ms")
            Text("Estimated tokens: ${it.estimatedTokens}")
            Text("Estimated speed: %.2f tokens/s".format(it.tokensPerSecond))
        } ?: Text("No benchmark result yet.")
        Text("The benchmark is intentionally lightweight so it can be repeated on your laptop without changing model configuration.")
    }
}

@Composable private fun MemoryPage(count: Long, refresh: () -> Unit) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Memory", style = MaterialTheme.typography.h5)
        Text("Stored experiences: $count")
        Button(onClick = refresh) { Text("Refresh") }
        Text("Agent experiences, facts and learned skills are stored separately from model weights. This follows Gallery's separation of runtime state from model files.")
    }
}

@Composable private fun ToolsPage() {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Tools", style = MaterialTheme.typography.h5)
        Text("Calculator")
        Text("Date/time")
        Text("Filesystem / process tools")
        Text("Web research")
        Text("Memory and learned skills")
        Text("Tools are described to the model, checked against permissions, executed independently, and their results are returned to the agent loop.")
    }
}

@Composable private fun LearningPage(learnFailures: Boolean, onFailures: (Boolean) -> Unit, skills: Boolean, onSkills: (Boolean) -> Unit, reflection: Boolean, onReflection: (Boolean) -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Learning", style = MaterialTheme.typography.h5)
        CheckRow("Learn from failed runs", learnFailures, onFailures)
        CheckRow("Learn reusable skills", skills, onSkills)
        CheckRow("Self-reflection / verification", reflection, onReflection)
        Text("Learning updates experience memory and skills; it never modifies Gemma weights.")
    }
}

@Composable private fun SettingsPage(mode: AgentMode, onMode: (AgentMode) -> Unit, maxIterations: Float, onIterations: (Float) -> Unit, memoryTopK: Float, onMemory: (Float) -> Unit, research: Boolean, onResearch: (Boolean) -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Settings", style = MaterialTheme.typography.h5)
        Text("Agent mode")
        AgentMode.values().forEach { m -> Row(verticalAlignment = Alignment.CenterVertically) { RadioButton(mode == m, { onMode(m) }); Text(m.name) } }
        Text("Max iterations: ${maxIterations.toInt()}")
        Slider(value = maxIterations, onValueChange = onIterations, valueRange = 1f..50f)
        Text("Memory retrieval: ${memoryTopK.toInt()}")
        Slider(value = memoryTopK, onValueChange = onMemory, valueRange = 0f..20f)
        CheckRow("Web research enabled", research, onResearch)
    }
}

@Composable private fun CheckRow(label: String, checked: Boolean, onChecked: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(checked, onChecked); Text(label) }
}
