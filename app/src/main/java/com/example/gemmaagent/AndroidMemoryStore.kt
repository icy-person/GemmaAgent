package com.example.gemmaagent

import android.content.Context
import com.example.gemmaagent.rust.RustMemory
import com.example.gemmaagent.shared.Experience
import com.example.gemmaagent.shared.MemoryFact
import com.example.gemmaagent.shared.MemoryStore
import com.example.gemmaagent.shared.Skill
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File

class AndroidMemoryStore(context: Context) : MemoryStore, AutoCloseable {
    private val dir = File(context.filesDir, "agent-memory").apply { mkdirs() }
    private val factsFile = File(dir, "facts.json")
    private val skillsFile = File(dir, "skills.json")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; prettyPrint = false }
    private val rust = RustMemory(dir.absolutePath)

    override suspend fun search(query: String, limit: Int): List<Experience> = withContext(Dispatchers.IO) {
        rust.search(query, limit)
    }

    override suspend fun store(experience: Experience) = withContext(Dispatchers.IO) {
        rust.store(experience)
        Unit
    }

    override suspend fun count(): Long = withContext(Dispatchers.IO) { rust.count() }

    override suspend fun saveFact(fact: MemoryFact) = withContext(Dispatchers.IO) {
        val current = readList(factsFile, ListSerializer(MemoryFact.serializer()))
            .filterNot { it.key == fact.key }
        factsFile.writeText(json.encodeToString(ListSerializer(MemoryFact.serializer()), current + fact.copy(updatedAtEpochMs = System.currentTimeMillis())))
    }

    override suspend fun searchFacts(query: String, limit: Int): List<MemoryFact> = withContext(Dispatchers.IO) {
        val terms = tokenize(query)
        readList(factsFile, ListSerializer(MemoryFact.serializer())).sortedByDescending {
            tokenize(it.key + " " + it.value).intersect(terms).size + (it.confidence * 0.1)
        }.take(limit)
    }

    override suspend fun saveSkill(skill: Skill) = withContext(Dispatchers.IO) {
        val current = readList(skillsFile, ListSerializer(Skill.serializer()))
        val existing = current.firstOrNull { it.name == skill.name }
        val merged = if (existing == null) skill else skill.copy(
            id = existing.id,
            useCount = existing.useCount + 1,
            successRate = ((existing.successRate * existing.useCount) + skill.successRate) / (existing.useCount + 1),
        )
        skillsFile.writeText(json.encodeToString(ListSerializer(Skill.serializer()), current.filterNot { it.name == skill.name } + merged))
    }

    override suspend fun searchSkills(query: String, limit: Int): List<Skill> = withContext(Dispatchers.IO) {
        val terms = tokenize(query)
        readList(skillsFile, ListSerializer(Skill.serializer())).sortedByDescending {
            tokenize(it.name + " " + it.description + " " + it.triggerTerms.joinToString(" ")).intersect(terms).size + it.successRate
        }.take(limit)
    }

    private fun <T> readList(file: File, serializer: kotlinx.serialization.KSerializer<List<T>>): List<T> =
        runCatching { if (!file.isFile) emptyList() else json.decodeFromString(serializer, file.readText()) }.getOrDefault(emptyList())

    private fun tokenize(s: String): Set<String> = s.lowercase().split(Regex("[^\\p{L}\\p{N}]+" )).filter { it.length > 2 }.toSet()

    override fun close() = rust.close()
}
