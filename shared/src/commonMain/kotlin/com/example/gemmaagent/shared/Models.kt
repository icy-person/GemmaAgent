package com.example.gemmaagent.shared

import kotlinx.serialization.Serializable

@Serializable
enum class AgentMode { SAFE, ASSISTED, AUTONOMOUS }

@Serializable
enum class Permission { READ, WRITE, EXECUTE, NETWORK, SYSTEM, NOTIFY, CLIPBOARD, CAMERA, MICROPHONE }

@Serializable
data class ToolDefinition(
    val name: String,
    val description: String,
    val category: String = "general",
    val permissions: Set<Permission> = setOf(Permission.READ),
    val dangerous: Boolean = false,
)

@Serializable
data class ToolCall(val name: String, val argumentsJson: String)

@Serializable
data class ToolResult(
    val ok: Boolean,
    val content: String,
    val durationMs: Long = 0,
    val metadata: Map<String, String> = emptyMap(),
)

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
    val durationMs: Long = 0,
    val project: String? = null,
    val tags: List<String> = emptyList(),
    val failureReason: String? = null,
)

@Serializable
data class Skill(
    val id: String = "",
    val name: String,
    val description: String,
    val triggerTerms: List<String> = emptyList(),
    val toolSequence: List<String> = emptyList(),
    val successRate: Double = 0.0,
    val useCount: Long = 0,
    val updatedAtEpochMs: Long = 0,
)

@Serializable
data class MemoryFact(
    val id: String = "",
    val key: String,
    val value: String,
    val confidence: Double = 0.5,
    val source: String = "agent",
    val createdAtEpochMs: Long = 0,
    val updatedAtEpochMs: Long = 0,
)

@Serializable
data class AgentConfig(
    val maxIterations: Int = 50,
    val maxContextChars: Int = 100_000,
    val memoryTopK: Int = 8,
    val skillTopK: Int = 5,
    val reflectionEnabled: Boolean = true,
    val learnFromFailures: Boolean = true,
    val mode: AgentMode = AgentMode.ASSISTED,
)

@Serializable
sealed class AgentEvent {
    @Serializable data class Started(val task: String, val atMs: Long) : AgentEvent()
    @Serializable data class Thinking(val iteration: Int) : AgentEvent()
    @Serializable data class ToolRequested(val iteration: Int, val call: ToolCall) : AgentEvent()
    @Serializable data class ToolCompleted(val iteration: Int, val call: ToolCall, val result: ToolResult) : AgentEvent()
    @Serializable data class Reflection(val text: String) : AgentEvent()
    @Serializable data class Finished(val success: Boolean, val answer: String) : AgentEvent()
    @Serializable data class Failed(val message: String) : AgentEvent()
}

data class AgentRun(
    val answer: String,
    val success: Boolean,
    val iterations: Int,
    val experienceId: String,
)

sealed interface ModelContent {
    data class Text(val text: String) : ModelContent
    data class ImageFile(val absolutePath: String) : ModelContent
    data class AudioFile(val absolutePath: String) : ModelContent
    data class ImageBytes(val bytes: ByteArray) : ModelContent
    data class AudioBytes(val bytes: ByteArray) : ModelContent
}

interface MemoryStore {
    suspend fun search(query: String, limit: Int = 5): List<Experience>
    suspend fun store(experience: Experience)
    suspend fun count(): Long
    suspend fun saveFact(fact: MemoryFact)
    suspend fun searchFacts(query: String, limit: Int = 10): List<MemoryFact>
    suspend fun saveSkill(skill: Skill)
    suspend fun searchSkills(query: String, limit: Int = 5): List<Skill>
}

interface ModelRunner {
    suspend fun generate(prompt: String): String
    suspend fun generate(contents: List<ModelContent>): String = error("Multimodal input is not supported by this model runner")
}

interface AgentTool {
    val definition: ToolDefinition
    suspend fun execute(argumentsJson: String): ToolResult
}

interface ToolApproval {
    suspend fun approve(definition: ToolDefinition, argumentsJson: String): Boolean
}

interface AgentObserver {
    fun onEvent(event: AgentEvent)
}
