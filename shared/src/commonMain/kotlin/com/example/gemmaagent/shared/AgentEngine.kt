package com.example.gemmaagent.shared

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlin.time.Duration.Companion.seconds

class AgentEngine(
    private val model: ModelRunner,
    private val memory: MemoryStore,
    tools: List<AgentTool>,
    private val maxIterations: Int = 50
) {
    private val toolsByName = tools.associateBy { it.name }
    private val mutex = Mutex()
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    suspend fun run(task: String): String = mutex.withLock {
        val memories = memory.search(task, 5)
        val history = mutableListOf<String>()
        val retrieved = buildString {
            for ((i, e) in memories.withIndex()) {
                appendLine("Experience ${i + 1}: success=${e.success}, score=${e.score}")
                appendLine("Task: ${e.task}")
                appendLine("Plan: ${e.plan}")
                appendLine("Result: ${e.result}")
            }
        }

        var prompt = systemPrompt() + "\n\nTASK:\n$task\n\nRELEVANT PAST EXPERIENCES:\n$retrieved"
        var finalAnswer = ""
        var success = false
        var lastPlan = ""

        repeat(maxIterations) { iteration ->
            val response = model.generate(prompt)
            history += response

            val action = parseAction(response)
            if (action == null) {
                finalAnswer = response
                success = true
                return@repeat
            }

            lastPlan = response
            val tool = toolsByName[action.name]
            val result = if (tool == null) {
                ToolResult(false, "Unknown tool: ${action.name}")
            } else {
                runCatching { tool.execute(action.argumentsJson) }
                    .getOrElse { ToolResult(false, "Tool exception: ${it.message}") }
            }

            prompt = buildString {
                append(systemPrompt())
                append("\n\nTASK:\n")
                append(task)
                append("\n\nPAST EXPERIENCES:\n")
                append(retrieved)
                append("\n\nITERATION: ")
                append(iteration + 1)
                append("\nMODEL RESPONSE:\n")
                append(response)
                append("\n\nTOOL RESULT:\n")
                append(result.content)
                append("\n\nReturn either a final answer or the next tool JSON action.")
            }
        }

        if (finalAnswer.isBlank()) {
            finalAnswer = "Stopped after $maxIterations iterations without a final answer."
        }

        val experience = Experience(
            task = task,
            plan = lastPlan,
            actions = history.takeLast(maxIterations),
            result = finalAnswer,
            success = success,
            score = if (success) 1.0 else 0.2,
            createdAtEpochMs = nowEpochMs(),
            tags = listOf("agent")
        )
        memory.store(experience)
        finalAnswer
    }

    private fun parseAction(text: String): ToolCall? {
        val start = text.lastIndexOf("{\"tool\"")
        if (start < 0) return null
        val end = text.indexOf('}', start)
        if (end < 0) return null
        val candidate = text.substring(start, end + 1)
        return runCatching { json.decodeFromString<ToolEnvelope>(candidate).asCall() }.getOrNull()
    }

    private fun systemPrompt(): String = """
        You are a local Android/Desktop agent.
        You have access to named tools. When you need a tool, emit exactly JSON like:
        {\"tool\":\"tool_name\",\"arguments\":{...}}
        When the task is complete, answer normally without a tool JSON object.
        Never invent tool results. Work iteratively and verify important actions.
    """.trimIndent()

    @kotlinx.serialization.Serializable
    private data class ToolEnvelope(val tool: String, val arguments: JsonObject = buildJsonObject { }) {
        fun asCall(): ToolCall = ToolCall(tool, arguments.toString())
    }
}

expect fun nowEpochMs(): Long
