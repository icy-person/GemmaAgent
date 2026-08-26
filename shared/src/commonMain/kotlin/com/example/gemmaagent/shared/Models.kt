package com.example.gemmaagent.shared

import kotlinx.serialization.Serializable

@Serializable
data class ToolCall(val name: String, val argumentsJson: String)

@Serializable
data class ToolResult(val ok: Boolean, val content: String)

@Serializable
data class Experience(
    val id: String = "",
    val task: String,
    val plan: String = "",
    val actions: List<String> = emptyList(),
    val result: String = "",
    val success: Boolean = false,
    val score: Double = 0.0,
    val createdAtEpochMs: Long = 0,
    val project: String? = null,
    val tags: List<String> = emptyList()
)

interface MemoryStore {
    suspend fun search(query: String, limit: Int = 5): List<Experience>
    suspend fun store(experience: Experience)
    suspend fun count(): Long
}

interface ModelRunner {
    suspend fun generate(prompt: String): String
}

interface AgentTool {
    val name: String
    val description: String
    suspend fun execute(argumentsJson: String): ToolResult
}
