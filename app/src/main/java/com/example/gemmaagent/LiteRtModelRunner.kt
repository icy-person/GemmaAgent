package com.example.gemmaagent

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.tool
import com.example.gemmaagent.shared.ModelContent
import com.example.gemmaagent.shared.ModelRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class LiteRtModelRunner(
    private val modelPath: String,
    private val cachePath: String,
    private val settings: ModelSettings,
    private val backend: Backend = Backend.CPU(),
) : ModelRunner, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var started = false

    suspend fun start(toolDefinitions: List<OpenApiTool>) = withContext(Dispatchers.IO) {
        val e = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = backend,
                cacheDir = cachePath,
                visionBackend = backend,
                audioBackend = Backend.CPU(),
            )
        )
        e.initialize()
        engine = e
        conversation = createConversation(e, toolDefinitions)
        started = true
    }

    private fun createConversation(e: Engine, toolDefinitions: List<OpenApiTool>): Conversation =
        e.createConversation(
            ConversationConfig(
                samplerConfig = SamplerConfig(
                    topK = settings.topK,
                    topP = settings.topP.toDouble(),
                    temperature = settings.temperature.toDouble(),
                ),
                automaticToolCalling = false,
                tools = toolDefinitions.map { tool(it) },
            )
        )

    override suspend fun reset() = withContext(Dispatchers.IO) {
        val e = engine ?: return@withContext
        conversation?.close()
        conversation = createConversation(e, emptyList())
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        check(started) { "Model is not started" }
        val c = conversation ?: error("Model conversation is unavailable")
        c.sendMessage(prompt).text
    }

    override suspend fun generate(contents: List<ModelContent>): String = withContext(Dispatchers.Default) {
        check(started) { "Model is not started" }
        val c = conversation ?: error("Model conversation is unavailable")
        val mapped = contents.map {
            when (it) {
                is ModelContent.Text -> Content.Text(it.text)
                is ModelContent.ImageFile -> Content.ImageFile(it.absolutePath)
                is ModelContent.AudioFile -> Content.AudioFile(it.absolutePath)
                is ModelContent.ImageBytes -> Content.ImageBytes(it.bytes)
                is ModelContent.AudioBytes -> Content.AudioBytes(it.bytes)
            }
        }
        c.sendMessage(Contents.of(mapped)).text
    }

    override fun close() {
        conversation?.close(); conversation = null
        engine?.close(); engine = null
        started = false
    }
}
