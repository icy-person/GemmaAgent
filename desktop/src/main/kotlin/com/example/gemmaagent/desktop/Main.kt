package com.example.gemmaagent.desktop

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.Text
import androidx.compose.material.TextField
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import com.example.gemmaagent.shared.AgentConfig
import com.example.gemmaagent.shared.AgentEngine
import com.example.gemmaagent.shared.CalculatorTool
import com.example.gemmaagent.shared.DateTimeTool
import com.example.gemmaagent.shared.EchoTool
import com.example.gemmaagent.shared.JvmMemoryStore
import com.example.gemmaagent.shared.ModelRunner
import com.example.gemmaagent.shared.platformTools
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

private class DesktopModelRunner(private val path: String) : ModelRunner, AutoCloseable {
    private val engine = Engine(EngineConfig(modelPath = path, backend = Backend.CPU()))
    private lateinit var conversation: com.google.ai.edge.litertlm.Conversation
    suspend fun start() = withContext(Dispatchers.IO) {
        engine.initialize()
        conversation = engine.createConversation(ConversationConfig(systemInstruction = Contents.of("You are GemmaAgent.")))
    }
    override suspend fun generate(prompt: String): String = conversation.sendMessage(prompt).text
    override fun close() { runCatching { conversation.close() }; runCatching { engine.close() } }
}

fun main() = application {
    Window(onCloseRequest = ::exitApplication, title = "GemmaAgent") {
        var path by remember { mutableStateOf("") }
        var task by remember { mutableStateOf("") }
        var answer by remember { mutableStateOf("") }
        var status by remember { mutableStateOf("Choose a .litertlm model path") }
        val scope = rememberCoroutineScope()
        var runner by remember { mutableStateOf<DesktopModelRunner?>(null) }
        var agent by remember { mutableStateOf<AgentEngine?>(null) }
        val memory = remember { JvmMemoryStore() }

        MaterialTheme {
            Column(Modifier.fillMaxSize().padding(16.dp)) {
                Text("GemmaAgent", style = MaterialTheme.typography.h5)
                Text(status)
                TextField(path, { path = it }, Modifier.fillMaxWidth(), label = { Text("Model path") })
                Row {
                    Button(onClick = {
                        scope.launch {
                            runCatching {
                                require(path.endsWith(".litertlm", true)) { "Model path must point to a .litertlm file" }
                                status = "Loading model..."
                                val r = DesktopModelRunner(path)
                                r.start()
                                runner?.close(); runner = r
                                val tools = listOf(CalculatorTool(), DateTimeTool(), EchoTool()) + platformTools(File(path).parent ?: ".")
                                agent = AgentEngine(r, memory, tools, AgentConfig())
                                status = "Model ready"
                            }.onFailure { status = "Load failed: ${it.message}" }
                        }
                    }) { Text("Load") }
                    OutlinedButton(onClick = { answer = "Memories: ${memory.count()}" }) { Text("Memory") }
                }
                TextField(task, { task = it }, Modifier.fillMaxWidth().padding(top = 8.dp), label = { Text("Task") })
                Button(enabled = agent != null && task.isNotBlank(), onClick = {
                    scope.launch { runCatching { agent!!.run(task).answer }.onSuccess { answer = it }.onFailure { answer = "Error: ${it.message}" } }
                }) { Text("Run Agent") }
                Text(answer, Modifier.fillMaxWidth().weight(1f).padding(top = 12.dp))
            }
        }
    }
}
