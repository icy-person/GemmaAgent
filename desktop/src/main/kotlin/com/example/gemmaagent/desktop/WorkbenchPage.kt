package com.example.gemmaagent.desktop

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.gemmaagent.shared.PluginRegistry
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject

@Composable
fun WorkbenchPage(registry: PluginRegistry) {
    val scope = rememberCoroutineScope()
    var filePath by remember { mutableStateOf("") }
    var archiveAction by remember { mutableStateOf("list") }
    var outputDir by remember { mutableStateOf("") }
    var buildAction by remember { mutableStateOf("build") }
    var buildSystem by remember { mutableStateOf("auto") }
    var gitAction by remember { mutableStateOf("status") }
    var gitRepo by remember { mutableStateOf("") }
    var gitBranch by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }

    fun runTool(name: String, args: String) {
        scope.launch {
            val tool = registry.allTools().firstOrNull { it.definition.name == name }
            result = if (tool == null) "Tool not installed: $name" else tool.execute(args).content
        }
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Workbench")
        Text("Files, archives, GitHub, repositories and local build systems.")
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("File inspection")
                OutlinedTextField(filePath, { filePath = it }, Modifier.fillMaxWidth(), label = { Text("Workspace-relative path") })
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { runTool("inspect_file", buildJsonObject { put("path", JsonPrimitive(filePath)) }.toString()) }) { Text("Inspect") }
                }
                Text("Archive")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(archiveAction, { archiveAction = it }, Modifier.weight(1f), label = { Text("list / extract") })
                    OutlinedTextField(outputDir, { outputDir = it }, Modifier.weight(1f), label = { Text("Output directory") })
                    Button(onClick = { runTool("archive", buildJsonObject {
                        put("action", JsonPrimitive(archiveAction)); put("path", JsonPrimitive(filePath)); put("output_dir", JsonPrimitive(outputDir))
                    }.toString()) }) { Text("Open") }
                }
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("GitHub / Git")
                OutlinedTextField(gitAction, { gitAction = it }, Modifier.fillMaxWidth(), label = { Text("status / clone / pull / branches / checkout / log / diff") })
                OutlinedTextField(gitRepo, { gitRepo = it }, Modifier.fillMaxWidth(), label = { Text("Repository URL for clone") })
                OutlinedTextField(gitBranch, { gitBranch = it }, Modifier.fillMaxWidth(), label = { Text("Branch") })
                Button(onClick = { runTool("git_workspace", buildJsonObject {
                    put("action", JsonPrimitive(gitAction)); put("repo", JsonPrimitive(gitRepo)); put("branch", JsonPrimitive(gitBranch)); put("destination", JsonPrimitive("repo"))
                }.toString()) }) { Text("Run Git action") }
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Build / Test / Lint")
                OutlinedTextField(buildAction, { buildAction = it }, Modifier.fillMaxWidth(), label = { Text("build / test / lint / clean / check") })
                OutlinedTextField(buildSystem, { buildSystem = it }, Modifier.fillMaxWidth(), label = { Text("auto / gradle / cargo / maven / npm / cmake") })
                Button(onClick = { runTool("build", buildJsonObject {
                    put("action", JsonPrimitive(buildAction)); put("system", JsonPrimitive(buildSystem))
                }.toString()) }) { Text("Execute") }
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Tool output")
                Text(result.ifBlank { "No operation executed yet." })
            }
        }
    }
}
