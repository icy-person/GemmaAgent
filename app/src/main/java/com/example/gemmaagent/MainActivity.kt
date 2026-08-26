package com.example.gemmaagent

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.gemmaagent.shared.AgentConfig
import com.example.gemmaagent.shared.AgentEngine
import com.example.gemmaagent.shared.AgentEvent
import com.example.gemmaagent.shared.AgentMode
import com.example.gemmaagent.shared.AgentObserver
import com.example.gemmaagent.shared.CalculatorTool
import com.example.gemmaagent.shared.DateTimeTool
import com.example.gemmaagent.shared.EchoTool
import com.example.gemmaagent.shared.platformTools
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AndroidAgentContext.context = applicationContext
        setContent { GemmaAgentApp() }
    }
}

enum class Panel { CHAT, MODEL, MEMORY, TOOLS, LEARNING, SETTINGS }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun GemmaAgentApp() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val repo = remember { ModelRepository(context) }
    val memory = remember { AndroidMemoryStore(context) }
    var settings by remember { mutableStateOf(ModelSettings.load(context)) }
    var modelPath by remember { mutableStateOf(repo.lastImportedPath()) }
    var runner by remember { mutableStateOf<LiteRtModelRunner?>(null) }
    var agent by remember { mutableStateOf<AgentEngine?>(null) }
    var prompt by remember { mutableStateOf("") }
    var answer by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("No model loaded") }
    var panel by remember { mutableStateOf(Panel.CHAT) }
    var modeMenu by remember { mutableStateOf(false) }
    var events by remember { mutableStateOf(listOf<String>()) }
    var memoryCount by remember { mutableStateOf(0L) }

    val observer = remember {
        object : AgentObserver {
            override fun onEvent(event: AgentEvent) {
                events = (events + event.toString()).takeLast(120)
            }
        }
    }

    fun createAgent(r: LiteRtModelRunner) {
        val workspace = context.filesDir.resolve("agent-workspace").absolutePath
        val tools = buildList {
            add(CalculatorTool())
            add(DateTimeTool())
            add(EchoTool())
            addAll(platformTools(workspace))
        }
        agent = AgentEngine(r, memory, tools, settings.agentConfig(), observer = observer)
    }

    suspend fun loadModel() {
        val path = modelPath ?: error("Import a model first")
        status = "Loading Gemma 4 E4B…"
        val r = LiteRtModelRunner(path, context.cacheDir.resolve("litertlm").absolutePath, settings)
        r.start(emptyList())
        runner?.close()
        runner = r
        createAgent(r)
        status = "Ready · ${File(path).name}"
    }

    val modelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            runCatching {
                status = "Importing model…"
                modelPath = repo.importModel(uri)
                loadModel()
            }.onFailure { status = "Import failed: ${it.message}" }
        }
    }

    val folderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            runCatching {
                status = "Importing model folder…"
                modelPath = repo.importFolder(uri)
                loadModel()
            }.onFailure { status = "Folder import failed: ${it.message}" }
        }
    }

    LaunchedEffect(Unit) { memoryCount = memory.count() }

    Surface(Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text("GemmaAgent", style = MaterialTheme.typography.headlineSmall)
                    Text(status, style = MaterialTheme.typography.bodySmall)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Button(onClick = { modelPicker.launch(arrayOf("application/octet-stream", "*/*")) }) { Text("Import") }
                    OutlinedButton(onClick = { folderPicker.launch(null) }) { Text("Folder") }
                }
            }

            ScrollableTabRow(selectedTabIndex = panel.ordinal) {
                Panel.values().forEach { p ->
                    Tab(selected = panel == p, onClick = { panel = p }, text = { Text(p.name.lowercase().replaceFirstChar { it.uppercase() }) })
                }
            }

            when (panel) {
                Panel.CHAT -> ChatPanel(
                    prompt = prompt,
                    answer = answer,
                    events = events,
                    mode = settings.mode,
                    modeMenu = modeMenu,
                    onModeMenu = { modeMenu = it },
                    onMode = { m ->
                        settings = settings.copy(mode = m)
                        settings.save(context)
                        agent = runner?.let { r -> createAgent(r); agent }
                    },
                    onPrompt = { prompt = it },
                    onRun = {
                        val a = agent ?: return@ChatPanel
                        scope.launch {
                            status = "Agent running…"
                            runCatching { a.run(prompt).also { answer = it.answer; memoryCount = memory.count() } }
                                .onFailure { answer = "Error: ${it.message}" }
                            status = "Ready · memories=$memoryCount"
                        }
                    },
                    enabled = agent != null && prompt.isNotBlank(),
                )
                Panel.MODEL -> ModelPanel(
                    modelPath = modelPath,
                    settings = settings,
                    onSettings = { settings = it },
                    onApply = {
                        settings.save(context)
                        scope.launch { runCatching { loadModel() }.onFailure { status = "Reload failed: ${it.message}" } }
                    },
                    onReset = {
                        settings = ModelSettings()
                        settings.save(context)
                    },
                    onUnload = {
                        runner?.close(); runner = null; agent = null; status = "Model unloaded"
                    },
                )
                Panel.MEMORY -> MemoryPanel(count = memoryCount, events = events)
                Panel.TOOLS -> ToolsPanel()
                Panel.LEARNING -> LearningPanel(events = events)
                Panel.SETTINGS -> SettingsPanel(settings = settings, onSettings = { settings = it; it.save(context) }, status = status)
            }
        }
    }
}

@Composable
private fun ChatPanel(
    prompt: String,
    answer: String,
    events: List<String>,
    mode: AgentMode,
    modeMenu: Boolean,
    onModeMenu: (Boolean) -> Unit,
    onMode: (AgentMode) -> Unit,
    onPrompt: (String) -> Unit,
    onRun: () -> Unit,
    enabled: Boolean,
) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = true, onClick = {}, label = { Text(mode.name) })
            OutlinedButton(onClick = { onModeMenu(true) }) { Text("Change mode") }
            DropdownMenu(expanded = modeMenu, onDismissRequest = { onModeMenu(false) }) {
                AgentMode.values().forEach { m -> DropdownMenuItem(text = { Text(m.name) }, onClick = { onMode(m); onModeMenu(false) }) }
            }
        }
        OutlinedTextField(prompt, onPrompt, Modifier.fillMaxWidth(), label = { Text("Task") }, minLines = 4)
        Button(onClick = onRun, enabled = enabled, modifier = Modifier.fillMaxWidth()) { Text("Run Agent") }
        Card(Modifier.fillMaxWidth().weight(1f)) {
            LazyColumn(Modifier.padding(12.dp)) { item { Text("Answer", style = MaterialTheme.typography.titleMedium); Text(answer) } }
        }
        Card(Modifier.fillMaxWidth().heightIn(max = 160.dp)) {
            LazyColumn(Modifier.padding(8.dp)) { items(events.takeLast(10)) { Text(it, style = MaterialTheme.typography.bodySmall) } }
        }
    }
}

@Composable
private fun ModelPanel(
    modelPath: String?,
    settings: ModelSettings,
    onSettings: (ModelSettings) -> Unit,
    onApply: () -> Unit,
    onReset: () -> Unit,
    onUnload: () -> Unit,
) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { Text("Model control", style = MaterialTheme.typography.headlineSmall) }
        item { Text("Loaded model: ${modelPath ?: "none"}") }
        item { Divider() }
        item { Text("Temperature: ${"%.2f".format(settings.temperature)}") }
        item { Slider(settings.temperature, { onSettings(settings.copy(temperature = it)) }, valueRange = 0f..2f) }
        item { OutlinedTextField(settings.topK.toString(), { it.toIntOrNull()?.let { v -> onSettings(settings.copy(topK = v.coerceIn(1, 256))) } }, label = { Text("Top-K") }) }
        item { Text("Top-P: ${"%.2f".format(settings.topP)}") }
        item { Slider(settings.topP, { onSettings(settings.copy(topP = it)) }, valueRange = 0.05f..1f) }
        item { OutlinedTextField(settings.maxIterations.toString(), { it.toIntOrNull()?.let { v -> onSettings(settings.copy(maxIterations = v.coerceIn(1, 200))) } }, label = { Text("Max Agent iterations") }) }
        item { OutlinedTextField(settings.maxContextChars.toString(), { it.toIntOrNull()?.let { v -> onSettings(settings.copy(maxContextChars = v.coerceIn(2_000, 500_000))) } }, label = { Text("Context character budget") }) }
        item { OutlinedTextField(settings.memoryTopK.toString(), { it.toIntOrNull()?.let { v -> onSettings(settings.copy(memoryTopK = v.coerceIn(0, 50))) } }, label = { Text("Memory retrieval Top-K") }) }
        item { OutlinedTextField(settings.skillTopK.toString(), { it.toIntOrNull()?.let { v -> onSettings(settings.copy(skillTopK = v.coerceIn(0, 25))) } }, label = { Text("Skill retrieval Top-K") }) }
        item { ToggleRow("Self verification / reflection", settings.reflectionEnabled) { onSettings(settings.copy(reflectionEnabled = it)) } }
        item { ToggleRow("Learn from failed runs", settings.learnFromFailures) { onSettings(settings.copy(learnFromFailures = it)) } }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onApply) { Text("Apply & Reload") }
                OutlinedButton(onClick = onReset) { Text("Reset defaults") }
                OutlinedButton(onClick = onUnload) { Text("Unload") }
            }
        }
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onChecked: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, Modifier.weight(1f))
        Checkbox(checked, onChecked)
    }
}

@Composable
private fun MemoryPanel(count: Long, events: List<String>) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Permanent memory", style = MaterialTheme.typography.headlineSmall)
        Text("Stored experiences: $count")
        Text("Memory is persistent and independent from Gemma weights.")
        Text("Recent learning events")
        LazyColumn(Modifier.weight(1f)) { items(events.takeLast(80)) { Text(it, style = MaterialTheme.typography.bodySmall) } }
    }
}

@Composable
private fun ToolsPanel() {
    val tools = listOf(
        "filesystem" to "Read/write/list inside the agent workspace",
        "http" to "GET/POST network requests",
        "calculator" to "Arithmetic and expressions",
        "datetime" to "Local date and time",
        "device" to "Android runtime/device operations",
        "clipboard" to "Copy text through Android clipboard adapter",
        "intent" to "Open Android URIs/intents through platform adapter",
    )
    LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        item { Text("Tools & permissions", style = MaterialTheme.typography.headlineSmall) }
        items(tools) { (name, desc) -> Card { Column(Modifier.padding(12.dp)) { Text(name, style = MaterialTheme.typography.titleMedium); Text(desc) } } }
    }
}

@Composable
private fun LearningPanel(events: List<String>) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Learning", style = MaterialTheme.typography.headlineSmall)
        Text("The model weights never change. The agent learns by storing successful and failed experiences and extracting reusable skills.")
        Text("Recent agent decisions")
        LazyColumn(Modifier.weight(1f)) { items(events.takeLast(100)) { Text(it, style = MaterialTheme.typography.bodySmall) } }
    }
}

@Composable
private fun SettingsPanel(settings: ModelSettings, onSettings: (ModelSettings) -> Unit, status: String) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { Text("Application settings", style = MaterialTheme.typography.headlineSmall) }
        item { Text("Status: $status") }
        item { ToggleRow("Reflection enabled", settings.reflectionEnabled) { onSettings(settings.copy(reflectionEnabled = it)) } }
        item { ToggleRow("Learn from failures", settings.learnFromFailures) { onSettings(settings.copy(learnFromFailures = it)) } }
        item { Text("Agent mode: ${settings.mode}") }
        item { Text("All settings are persisted locally on the device.") }
        item { Text("Model remains outside the APK and is loaded from the imported model file.") }
    }
}
