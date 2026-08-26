package com.example.gemmaagent

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.tool
import com.example.gemmaagent.shared.ModelRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class LiteRtModelRunner(
    private val modelPath: String,
    private val cachePath: String,
    private val backend: Backend = Backend.CPU(),
) : ModelRunner, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null

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
        conversation = e.createConversation(
            ConversationConfig(
                samplerConfig = SamplerConfig(topK = 64, topP = 0.95, temperature = 0.7),
                automaticToolCalling = false,
                tools = toolDefinitions.map { tool(it) },
            )
        )
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        val c = conversation ?: error("Model is not started")
        c.sendMessage(prompt).text
    }

    override fun close() {
        conversation?.close(); conversation = null
        engine?.close(); engine = null
    }
}
