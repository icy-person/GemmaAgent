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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
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
import com.example.gemmaagent.shared.AndroidAgentContext
import com.example.gemmaagent.shared.CalculatorTool
import com.example.gemmaagent.shared.DateTimeTool
import com.example.gemmaagent.shared.EchoTool
import com.example.gemmaagent.shared.platformTools
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AndroidAgentContext.context = applicationContext
        setContent { GemmaAgentApp() }
    }
}

@androidx.compose.runtime.Composable
private fun GemmaAgentApp() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val repo = remember { ModelRepository(context) }
    val memory = remember { AndroidMemoryStore(context) }
    val runnerHolder = remember { mutableStateOf<LiteRtModelRunner?>(null) }
    val agentHolder = remember { mutableStateOf<AgentEngine?>(null) }
    var modelPath by remember { mutableStateOf(repo.lastImportedPath()) }
    var prompt by remember { mutableStateOf("") }
    var answer by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("Model not loaded") }
    var mode by remember { mutableStateOf(AgentMode.ASSISTED) }
    var showMode by remember { mutableStateOf(false) }
    val events = remember { mutableStateOf(listOf<String>()) }

    val observer = remember {
        object : AgentObserver {
            override fun onEvent(event: AgentEvent) {
                events.value = (events.value + event.toString()).takeLast(80)
            }
        }
    }

    fun buildAgent(runner: LiteRtModelRunner) {
        val tools = buildList {
            add(CalculatorTool())
            add(DateTimeTool())
            add(EchoTool())
            addAll(platformTools(context.filesDir.resolve("agent-workspace").absolutePath))
        }
        agentHolder.value = AgentEngine(
            model = runner,
            memory = memory,
            tools = tools,
            config = AgentConfig(mode = mode),
            observer = observer,
        )
    }

    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            runCatching {
                status = "Importing model..."
                modelPath = repo.importModel(uri)
                status = "Imported: ${modelPath?.substringAfterLast('/') }"
            }.onFailure { status = "Import failed: ${it.message}" }
        }
    }

    val folderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            runCatching {
                status = "Searching model folder..."
                modelPath = repo.importFolder(uri)
                status = "Imported: ${modelPath?.substringAfterLast('/') }"
            }.onFailure { status = "Folder import failed: ${it.message}" }
        }
    }

    LaunchedEffect(modelPath) {
        val path = modelPath ?: return@LaunchedEffect
        status = "Loading Gemma 4 E4B..."
        runCatching {
            val runner = LiteRtModelRunner(path, context.cacheDir.resolve("litertlm").absolutePath)
            runner.start(emptyList())
            runnerHolder.value?.close()
            runnerHolder.value = runner
            buildAgent(runner)
            status = "Gemma 4 E4B ready"
        }.onFailure { status = "Load failed: ${it.message}" }
    }

    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(
            modifier = Modifier.fillMaxSize().padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("GemmaAgent", style = MaterialTheme.typography.headlineMedium)
            Text(status, style = MaterialTheme.typography.bodyMedium)

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { filePicker.launch(arrayOf("application/octet-stream", "*/*")) }) { Text("Import model") }
                OutlinedButton(onClick = { folderPicker.launch(null) }) { Text("Import folder") }
                OutlinedButton(onClick = { showMode = true }) { Text(mode.name) }
                DropdownMenu(expanded = showMode, onDismissRequest = { showMode = false }) {
                    AgentMode.values().forEach { m ->
                        DropdownMenuItem(text = { Text(m.name) }, onClick = { mode = m; showMode = false; runnerHolder.value?.let(::buildAgent) })
                    }
                }
            }

            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Task") },
                minLines = 3,
            )

            Button(
                onClick = {
                    val agent = agentHolder.value ?: return@Button
                    scope.launch {
                        status = "Agent running..."
                        runCatching { agent.run(prompt).also { answer = it.answer } }
                            .onFailure { answer = "Error: ${it.message}" }
                        status = "Ready · memory=${memory.count()}"
                    }
                },
                enabled = agentHolder.value != null && prompt.isNotBlank(),
            ) { Text("Run Agent") }

            Text("Answer", style = MaterialTheme.typography.titleMedium)
            Text(answer, modifier = Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState()))
            Text("Agent events", style = MaterialTheme.typography.titleMedium)
            Text(events.value.takeLast(12).joinToString("\n"), modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()))
        }
    }
}
