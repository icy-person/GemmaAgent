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
import androidx.compose.material.Divider
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
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
import com.example.gemmaagent.desktop.plugins.installDefaultPlugins
import com.example.gemmaagent.shared.AgentConfig
import com.example.gemmaagent.shared.AgentEngine
import com.example.gemmaagent.shared.AgentEvent
import com.example.gemmaagent.shared.AgentMetrics
import com.example.gemmaagent.shared.AgentMode
import com.example.gemmaagent.shared.AgentObserver
import com.example.gemmaagent.shared.JvmMemoryStore
import com.example.gemmaagent.shared.JvmRagStore
import com.example.gemmaagent.shared.PluginRegistry
import com.example.gemmaagent.shared.RagDocument
import com.example.gemmaagent.shared.RagEngine
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

private class AppModelRunner(private val path: String) : com.example.gemmaagent.shared.ModelRunner, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    suspend fun load() = withContext(Dispatchers.IO) {
        require(File(path).isFile) { "Model file not found" }
        require(path.endsWith(".litertlm", true)) { "Expected .litertlm" }
        closeResources()
        val e = Engine(EngineConfig(modelPath = path, backend = Backend.CPU()))
        try {
            e.initialize()
            engine = e
            reset()
        } catch (t: Throwable) {
            runCatching { e.close() }
            throw t
        }
    }

    override suspend fun reset() = withContext(Dispatchers.IO) {
        val e = engine ?: error("Model is not loaded")
        conversation?.close()
        conversation = e.createConversation(ConversationConfig(systemInstruction = Contents.of("You are GemmaAgent. Use tools when useful, verify tool results, use retrieved evidence when supplied, and never invent citations.")))
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        conversation?.sendMessage(prompt)?.toString() ?: error("Conversation unavailable")
    }

    override fun close() = closeResources()

    private fun closeResources() {
        runCatching { conversation?.close() }
        runCatching { engine?.close() }
        conversation = null
        engine = null
    }
}

private enum class Page { HOME, AGENT, RESEARCH, MODEL, WORKBENCH, RAG, PLUGINS, MEMORY, SETTINGS, LOGS }

fun main() = application {
    Window(onCloseRequest = ::exitApplication, title = "GemmaAgent") {
        val dataDir = remember { File(System.getProperty("user.home"), ".gemmaagent").apply { mkdirs() } }
        val registry = remember { PluginRegistry() }
        val memory = remember { JvmMemoryStore() }
        val ragStore = remember { JvmRagStore(File(dataDir, "rag/index.json")) }
        val rag = remember { RagEngine(ragStore) }
        val metrics = remember { AgentMetrics() }
        val scope = rememberCoroutineScope()
        var page by remember { mutableStateOf(Page.HOME) }
        var modelPath by remember { mutableStateOf("") }
        var model by remember { mutableStateOf<AppModelRunner?>(null) }
        var agent by remember { mutableStateOf<AgentEngine?>(null) }
        var status by remember { mutableStateOf("Starting") }
        var task by remember { mutableStateOf("") }
        var answer by remember { mutableStateOf("") }
        var ragQuery by remember { mutableStateOf("") }
        var ragFile by remember { mutableStateOf("") }
        var ragResult by remember { mutableStateOf("") }
        var mode by remember { mutableStateOf(AgentMode.ASSISTED) }
        var maxIterations by remember { mutableStateOf(30f) }
        var memoryTopK by remember { mutableStateOf(8f) }
        var events by remember { mutableStateOf(listOf<String>()) }
        val observer = remember { object : AgentObserver {
            override fun onEvent(event: AgentEvent) {
                metrics.onEvent(event)
                events = (events + event.toString()).takeLast(150)
            }
        } }

        fun rebuildAgent() {
            val runner = model ?: return
            scope.launch {
                val tools = registry.allTools()
                agent = AgentEngine(model = runner, memory = memory, tools = tools, config = AgentConfig(maxIterations = maxIterations.toInt(), memoryTopK = memoryTopK.toInt(), skillTopK = 5, reflectionEnabled = true, learnFromFailures = true, mode = mode), observer = observer)
                status = "Agent ready (${tools.size} tools)"
            }
        }

        scope.launch {
            runCatching {
                installDefaultPlugins(registry, dataDir)
                status = "Plugins ready"
                rebuildAgent()
            }.onFailure { status = "Plugin error: ${it.message}" }
        }

        val pages = Page.values().toList()
        val topPages = pages.take(6)
        MaterialTheme {
            Row(Modifier.fillMaxSize()) {
                Column(Modifier.width(220.dp).fillMaxHeight().padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("GemmaAgent", style = MaterialTheme.typography.h5)
                    Text(status)
                    Divider()
                    pages.forEach { item -> Button(onClick = { page = item }, Modifier.fillMaxWidth()) { Text(item.name.replace('_', ' ')) } }
                }
                Column(Modifier.fillMaxSize().padding(14.dp)) {
                    TabRow(selectedTabIndex = topPages.indexOf(page).coerceAtLeast(0)) {
                        topPages.forEach { item -> Tab(page == item, { page = item }, text = { Text(item.name.take(8)) }) }
                    }
                    Spacer(Modifier.height(12.dp))
                    when (page) {
                        Page.HOME -> HomePage(status, modelPath, agent != null, metrics.snapshot())
                        Page.AGENT -> AgentPage(task, { task = it }, answer, agent != null) { scope.launch { runCatching { agent!!.run(task).answer }.onSuccess { answer = it; status = "Task completed" }.onFailure { answer = "Error: ${it.message}"; status = "Task failed" } } }
                        Page.RESEARCH -> AgentPage(task, { task = it }, answer, agent != null) { scope.launch { runCatching { agent!!.run("Research this topic thoroughly on the public web, use the web research plugin, and produce a cited report: $task").answer }.onSuccess { answer = it; status = "Research completed" }.onFailure { answer = "Research error: ${it.message}" } } }
                        Page.MODEL -> ModelPage(modelPath, { modelPath = it }, status, {
                            val chooser = JFileChooser().apply { dialogTitle = "Select Gemma 4 E4B .litertlm" }
                            if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                                modelPath = chooser.selectedFile.absolutePath
                                scope.launch {
                                    runCatching {
                                        status = "Loading model..."
                                        val next = AppModelRunner(modelPath)
                                        next.load()
                                        model?.close()
                                        model = next
                                        rebuildAgent()
                                        status = "Gemma 4 E4B ready"
                                    }.onFailure { status = "Load failed: ${it.message}" }
                                }
                            }
                        }, { model?.close(); model = null; agent = null; status = "Model unloaded" })
                        Page.WORKBENCH -> WorkbenchPage(registry)
                        Page.RAG -> RagPage(rag, ragQuery, { ragQuery = it }, ragFile, { ragFile = it }, ragResult, { ragResult = it })
                        Page.PLUGINS -> PluginPage(registry)
                        Page.MEMORY -> MemoryPage(memory)
                        Page.SETTINGS -> SettingsPage(mode, { mode = it; rebuildAgent() }, maxIterations, { maxIterations = it; rebuildAgent() }, memoryTopK, { memoryTopK = it; rebuildAgent() })
                        Page.LOGS -> LogsPage(metrics.snapshot(), events)
                    }
                }
            }
        }
    }
}

@Composable private fun HomePage(status: String, modelPath: String, agentReady: Boolean, metrics: Map<String, Long>) = Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Overview", style = MaterialTheme.typography.h5); Text(status); Text("Model: ${modelPath.ifBlank { "not selected" }}"); Text("Agent: ${if (agentReady) "ready" else "not ready"}"); Text("Metrics: $metrics") }
@Composable private fun AgentPage(task: String, onTask: (String) -> Unit, answer: String, enabled: Boolean, run: () -> Unit) = Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Agent", style = MaterialTheme.typography.h5); OutlinedTextField(task, onTask, Modifier.fillMaxWidth(), label = { Text("Task") }, minLines = 5); Button(onClick = run, enabled = enabled && task.isNotBlank()) { Text("Run") }; Card(Modifier.fillMaxSize()) { Text(answer, Modifier.padding(14.dp).verticalScroll(rememberScrollState())) } }
@Composable private fun ModelPage(path: String, onPath: (String) -> Unit, status: String, load: () -> Unit, unload: () -> Unit) = Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Model", style = MaterialTheme.typography.h5); Text(status); OutlinedTextField(path, onPath, Modifier.fillMaxWidth(), label = { Text(".litertlm") }); Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { Button(onClick = load) { Text("Load") }; OutlinedButton(onClick = unload) { Text("Unload") } } }
@Composable private fun RagPage(rag: RagEngine, query: String, onQuery: (String) -> Unit, file: String, onFile: (String) -> Unit, result: String, onResult: (String) -> Unit) { val scope = rememberCoroutineScope(); Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text("Local RAG", style = MaterialTheme.typography.h5); Text("Persistent document retrieval without changing Gemma weights."); OutlinedTextField(file, onFile, Modifier.fillMaxWidth(), label = { Text("Text/Markdown file path") }); Button(onClick = { scope.launch { runCatching { val f = File(file); require(f.isFile) { "File not found" }; rag.ingest(RagDocument(f.name, f.absolutePath, f.name, f.readText())); onResult("Indexed ${f.name}") }.onFailure { onResult("RAG error: ${it.message}") } } }) { Text("Index file") }; OutlinedTextField(query, onQuery, Modifier.fillMaxWidth(), label = { Text("Query") }); Button(onClick = { scope.launch { onResult(rag.context(query)) } }, enabled = query.isNotBlank()) { Text("Retrieve") }; Card(Modifier.fillMaxWidth()) { Text(result, Modifier.padding(12.dp)) } } }
@Composable private fun PluginPage(registry: PluginRegistry) { val scope = rememberCoroutineScope(); var text by remember { mutableStateOf("") }; Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) { Button(onClick = { scope.launch { text = registry.all().joinToString("\n") { "${it.manifest.name} ${it.manifest.version}: ${it.manifest.description}" } } }) { Text("Refresh plugins") }; Text(text) } }
@Composable private fun MemoryPage(memory: JvmMemoryStore) { val scope = rememberCoroutineScope(); var text by remember { mutableStateOf("") }; Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(8.dp)) { Button(onClick = { scope.launch { text = "Experiences: ${memory.count()}" } }) { Text("Refresh") }; Text(text) } }
@Composable private fun SettingsPage(mode: AgentMode, onMode: (AgentMode) -> Unit, iterations: Float, onIterations: (Float) -> Unit, memoryTopK: Float, onMemory: (Float) -> Unit) = Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) { Text("Settings", style = MaterialTheme.typography.h5); AgentMode.values().forEach { m -> Row(verticalAlignment = Alignment.CenterVertically) { androidx.compose.material.RadioButton(mode == m, { onMode(m) }); Text(m.name) } }; Text("Max iterations: ${iterations.toInt()}"); androidx.compose.material.Slider(iterations, onIterations, valueRange = 1f..50f); Text("RAG/Memory top-K: ${memoryTopK.toInt()}"); androidx.compose.material.Slider(memoryTopK, onMemory, valueRange = 0f..20f) }
@Composable private fun LogsPage(metrics: Map<String, Long>, events: List<String>) = Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) { Text("Metrics: $metrics"); Text(events.joinToString("\n")) }
