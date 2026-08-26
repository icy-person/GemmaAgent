package com.example.gemmaagent.shared

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlin.math.max

class AgentEngine(
    private val model: ModelRunner,
    private val memory: MemoryStore,
    tools: List<AgentTool>,
    private val config: AgentConfig = AgentConfig(),
    private val approval: ToolApproval = object : ToolApproval {
        override suspend fun approve(definition: ToolDefinition, argumentsJson: String): Boolean = !definition.dangerous
    },
    private val observer: AgentObserver = object : AgentObserver {
        override fun onEvent(event: AgentEvent) = Unit
    },
) {
    private val toolsByName = tools.associateBy { it.definition.name }
    private val mutex = Mutex()
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }

    suspend fun run(task: String, project: String? = null): AgentRun = mutex.withLock {
        val started = nowEpochMs()
        observer.onEvent(AgentEvent.Started(task, started))

        val memories = memory.search(task, config.memoryTopK)
        val facts = memory.searchFacts(task, 8)
        val skills = memory.searchSkills(task, config.skillTopK)
        val history = mutableListOf<String>()
        val toolHistory = mutableListOf<String>()
        var lastPlan = ""
        var finalAnswer = ""
        var success = false
        var iterationCount = 0

        var prompt = buildPrompt(task, project, memories, facts, skills, history, "")

        for (iteration in 1..config.maxIterations) {
            iterationCount = iteration
            observer.onEvent(AgentEvent.Thinking(iteration))
            val response = runCatching { model.generate(prompt) }
                .getOrElse {
                    observer.onEvent(AgentEvent.Failed(it.message ?: "Model error"))
                    break
                }
            history += "MODEL[$iteration]: $response"

            val action = parseAction(response)
            if (action == null) {
                finalAnswer = response.trim()
                success = finalAnswer.isNotBlank()
                if (config.reflectionEnabled && success) {
                    val reflectionPrompt = buildReflectionPrompt(task, finalAnswer, toolHistory)
                    val reflection = runCatching { model.generate(reflectionPrompt) }.getOrDefault("")
                    if (reflection.isNotBlank()) observer.onEvent(AgentEvent.Reflection(reflection))
                    if (reflection.contains("RETRY", ignoreCase = true)) {
                        prompt = buildPrompt(task, project, memories, facts, skills, history, "Reflection requested another attempt.")
                        continue
                    }
                }
                break
            }

            lastPlan = response
            val tool = toolsByName[action.name]
            if (tool == null) {
                val result = ToolResult(false, "Unknown tool: ${action.name}")
                observer.onEvent(AgentEvent.ToolCompleted(iteration, action, result))
                prompt = nextPrompt(task, project, memories, facts, skills, history, toolHistory, result)
                continue
            }

            observer.onEvent(AgentEvent.ToolRequested(iteration, action))
            val allowed = permissionAllowed(tool.definition) && approval.approve(tool.definition, action.argumentsJson)
            val result = if (!allowed) {
                ToolResult(false, "Permission denied for ${tool.definition.name}")
            } else {
                val begin = nowEpochMs()
                runCatching { tool.execute(action.argumentsJson) }
                    .getOrElse { ToolResult(false, "Tool exception: ${it.message}") }
                    .copy(durationMs = max(0, nowEpochMs() - begin))
            }
            toolHistory += "${action.name}: ${result.ok}"
            history += "TOOL[$iteration] ${action.name}: ${result.content}"
            observer.onEvent(AgentEvent.ToolCompleted(iteration, action, result))
            prompt = nextPrompt(task, project, memories, facts, skills, history, toolHistory, result)

            if (!result.ok && config.reflectionEnabled) {
                val reflection = runCatching {
                    model.generate(buildReflectionPrompt(task, "Tool ${action.name} failed: ${result.content}", toolHistory))
                }.getOrDefault("")
                if (reflection.isNotBlank()) observer.onEvent(AgentEvent.Reflection(reflection))
            }
        }

        if (finalAnswer.isBlank()) {
            finalAnswer = "Agent stopped after $iterationCount iterations without a final answer."
        }
        val duration = nowEpochMs() - started
        val score = scoreRun(success, toolHistory, iterationCount)
        val experience = Experience(
            id = stableId(task, started),
            task = task,
            plan = lastPlan,
            actions = toolHistory.toList(),
            result = finalAnswer,
            success = success,
            score = score,
            createdAtEpochMs = started,
            durationMs = duration,
            project = project,
            tags = listOf("agent", if (success) "success" else "failure"),
            failureReason = if (success) null else finalAnswer,
        )
        if (success || config.learnFromFailures) memory.store(experience)
        if (success && toolHistory.size >= 2) learnSkill(task, toolHistory, score)
        observer.onEvent(AgentEvent.Finished(success, finalAnswer))
        AgentRun(finalAnswer, success, iterationCount, experience.id)
    }

    private suspend fun learnSkill(task: String, toolHistory: List<String>, score: Double) {
        val sequence = toolHistory.map { it.substringBefore(":") }.distinct()
        if (sequence.size < 2) return
        val name = sequence.joinToString("_").lowercase().take(48)
        val skill = Skill(
            id = stableId(name, nowEpochMs()),
            name = name,
            description = "Learned workflow for tasks similar to: ${task.take(120)}",
            triggerTerms = task.lowercase().split(Regex("[^\\p{L}\\p{N}]+"))
                .filter { it.length >= 4 }.distinct().take(12),
            toolSequence = sequence,
            successRate = score,
            useCount = 1,
            updatedAtEpochMs = nowEpochMs(),
        )
        memory.saveSkill(skill)
    }

    private fun permissionAllowed(def: ToolDefinition): Boolean = when (config.mode) {
        AgentMode.SAFE -> def.permissions.all { it == Permission.READ }
        AgentMode.ASSISTED -> true
        AgentMode.AUTONOMOUS -> true
    }

    private fun buildPrompt(
        task: String,
        project: String?,
        memories: List<Experience>,
        facts: List<MemoryFact>,
        skills: List<Skill>,
        history: List<String>,
        extra: String,
    ): String {
        val toolSpec = toolsByName.values.joinToString("\n") {
            "- ${it.definition.name}: ${it.definition.description} [${it.definition.permissions.joinToString()}]"
        }
        return trimContext("""
            You are GemmaAgent, a local autonomous agent.
            Follow this loop: understand → plan → act with tools → inspect observations → verify → answer.
            Never invent tool results. Prefer small, reversible actions. If a requested tool is unavailable, explain it.
            Tool call format must be exactly JSON: {"tool":"tool_name","arguments":{...}}
            A normal response without a tool JSON means the task is complete.

            MODE: ${config.mode}
            PROJECT: ${project ?: "none"}
            TASK:
            $task

            AVAILABLE TOOLS:
            $toolSpec

            RELEVANT EXPERIENCES:
            ${renderExperiences(memories)}

            RELEVANT FACTS:
            ${facts.joinToString("\n") { "- ${it.key}=${it.value} (confidence=${it.confidence})" }}

            LEARNED SKILLS:
            ${skills.joinToString("\n") { "- ${it.name}: ${it.description}; sequence=${it.toolSequence.joinToString(" → ")}" }}

            RUN HISTORY:
            ${history.takeLast(14).joinToString("\n")}

            CONTROLLER NOTE:
            $extra
        """.trimIndent())
    }

    private fun nextPrompt(
        task: String,
        project: String?,
        memories: List<Experience>,
        facts: List<MemoryFact>,
        skills: List<Skill>,
        history: List<String>,
        toolHistory: List<String>,
        result: ToolResult,
    ): String = buildPrompt(
        task, project, memories, facts, skills, history,
        "Last tool result: ok=${result.ok}, duration=${result.durationMs}ms, content=${result.content}\n" +
            "Actions so far: ${toolHistory.joinToString(", ")}. Verify before finishing."
    )

    private fun buildReflectionPrompt(task: String, answer: String, actions: List<String>): String = """
        Act as a strict verifier for an agent run.
        Task: $task
        Proposed result: $answer
        Actions: ${actions.joinToString(", ")}
        Reply with one of:
        PASS - the answer is sufficiently verified.
        RETRY - more work or a tool check is required.
        Then one short reason.
    """.trimIndent()

    private fun renderExperiences(items: List<Experience>): String = if (items.isEmpty()) "none" else items.joinToString("\n") {
        "- success=${it.success}, score=${"%.2f".format(it.score)}, task=${it.task.take(180)}, plan=${it.plan.take(240)}, result=${it.result.take(240)}"
    }

    private fun trimContext(text: String): String = if (text.length <= config.maxContextChars) text else text.take(config.maxContextChars) + "\n[context trimmed]"

    private fun parseAction(text: String): ToolCall? {
        val start = text.lastIndexOf("{\"tool\"")
        if (start < 0) return null
        var depth = 0
        var end = -1
        var quoted = false
        var escaped = false
        for (i in start until text.length) {
            val c = text[i]
            if (escaped) { escaped = false; continue }
            if (c == '\\' && quoted) { escaped = true; continue }
            if (c == '"') quoted = !quoted
            if (!quoted) {
                if (c == '{') depth++
                if (c == '}') { depth--; if (depth == 0) { end = i; break } }
            }
        }
        if (end < 0) return null
        val candidate = text.substring(start, end + 1)
        return runCatching { json.decodeFromString<ToolEnvelope>(candidate).asCall() }.getOrNull()
    }

    private fun scoreRun(success: Boolean, actions: List<String>, iterations: Int): Double {
        if (!success) return if (actions.isEmpty()) 0.05 else 0.20
        val efficiency = (1.0 - ((iterations - 1).coerceAtLeast(0) / config.maxIterations.toDouble())).coerceIn(0.0, 1.0)
        val verification = if (actions.any { it.contains("verify", true) || it.contains("check", true) }) 0.1 else 0.0
        return (0.75 + efficiency * 0.15 + verification).coerceIn(0.0, 1.0)
    }

    private fun stableId(seed: String, salt: Long): String = (seed.hashCode().toLong() xor salt).toString(16)

    @Serializable
    private data class ToolEnvelope(val tool: String, val arguments: JsonObject = JsonObject(emptyMap())) {
        fun asCall(): ToolCall = ToolCall(tool, arguments.toString())
    }
}

expect fun nowEpochMs(): Long
