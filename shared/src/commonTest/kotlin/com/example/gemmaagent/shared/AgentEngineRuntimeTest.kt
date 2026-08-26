package com.example.gemmaagent.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AgentEngineRuntimeTest {
    private class Memory : MemoryStore {
        val experiences = mutableListOf<Experience>()
        val skills = mutableListOf<Skill>()
        override suspend fun search(query: String, limit: Int) = experiences.take(limit)
        override suspend fun store(experience: Experience) { experiences.removeAll { it.id == experience.id }; experiences += experience }
        override suspend fun count() = experiences.size.toLong()
        override suspend fun saveFact(fact: MemoryFact) = Unit
        override suspend fun searchFacts(query: String, limit: Int) = emptyList<MemoryFact>()
        override suspend fun saveSkill(skill: Skill) { skills += skill }
        override suspend fun searchSkills(query: String, limit: Int) = skills.take(limit)
    }

    private class Model : ModelRunner {
        private var index = 0
        override suspend fun reset() { index = 0 }
        override suspend fun generate(prompt: String): String = when (index++) {
            0 -> "{\"tool\":\"test_tool\",\"arguments\":{\"value\":\"ok\"}}"
            else -> "done"
        }
    }

    @Test
    fun toolLoopStoresSuccessfulExperience() = kotlinx.coroutines.test.runTest {
        val memory = Memory()
        var calls = 0
        val tool = object : AgentTool {
            override val definition = ToolDefinition("test_tool", "Test runtime tool")
            override suspend fun execute(argumentsJson: String): ToolResult {
                calls++
                return ToolResult(true, "tool-result")
            }
        }
        val engine = AgentEngine(
            model = Model(),
            memory = memory,
            tools = listOf(tool),
            config = AgentConfig(maxIterations = 3, reflectionEnabled = false),
        )

        val result = engine.run("test task")

        assertTrue(result.success)
        assertEquals("done", result.answer)
        assertEquals(1, calls)
        assertEquals(1, memory.experiences.size)
        assertTrue(memory.experiences.single().success)
    }

    @Test
    fun blankTaskIsRejected() = kotlinx.coroutines.test.runTest {
        val engine = AgentEngine(Model(), Memory(), emptyList())
        var rejected = false
        try {
            engine.run("   ")
        } catch (_: IllegalArgumentException) {
            rejected = true
        }
        assertTrue(rejected)
    }
}
