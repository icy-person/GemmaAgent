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
    @Volatile private var started = false

    suspend fun start(toolDefinitions: List<OpenApiTool>) = withContext(Dispatchers.IO) {
        closeInternal()
        val newEngine = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = backend,
                cacheDir = cachePath,
                visionBackend = backend,
                audioBackend = Backend.CPU(),
            )
        )
        try {
            newEngine.initialize()
            val newConversation = createConversation(newEngine, toolDefinitions)
            engine = newEngine
            conversation = newConversation
            started = true
        } catch (t: Throwable) {
            runCatching { newEngine.close() }
            started = false
            throw t
        }
    }

    private fun createConversation(e: Engine, toolDefinitions: List<OpenApiTool>): Conversation =
        e.createConversation(
            ConversationConfig(
                samplerConfig = SamplerConfig(
                    topK = settings.topK.coerceAtLeast(1),
                    topP = settings.topP.toDouble().coerceIn(0.01, 1.0),
                    temperature = settings.temperature.toDouble().coerceIn(0.0, 2.0),
                ),
                automaticToolCalling = false,
                tools = toolDefinitions.map { tool(it) },
            )
        )

    override suspend fun reset() = withContext(Dispatchers.IO) {
        check(started) { "Model is not started" }
        val e = engine ?: error("Model engine is unavailable")
        conversation?.let { runCatching { it.close() } }
        conversation = createConversation(e, emptyList())
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        require(prompt.isNotBlank()) { "Prompt must not be blank" }
        check(started) { "Model is not started" }
        val c = conversation ?: error("Model conversation is unavailable")
        c.sendMessage(prompt).toString()
    }

    override suspend fun generate(contents: List<ModelContent>): String = withContext(Dispatchers.Default) {
        require(contents.isNotEmpty()) { "Model contents must not be empty" }
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
        c.sendMessage(Contents.of(mapped)).toString()
    }

    private fun closeInternal() {
        started = false
        conversation?.let { runCatching { it.close() } }
        conversation = null
        engine?.let { runCatching { it.close() } }
        engine = null
    }

    override fun close() = synchronized(this) { closeInternal() }
}
