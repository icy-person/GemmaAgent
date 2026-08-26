package com.example.gemmaagent.shared

import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File

class JvmMemoryStore(private val root: File = File(System.getProperty("user.home"), ".gemma-agent/memory")) : MemoryStore {
    private val experiences = File(root, "experiences.json")
    private val facts = File(root, "facts.json")
    private val skills = File(root, "skills.json")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; prettyPrint = false }

    init { root.mkdirs() }

    @Synchronized
    override suspend fun search(query: String, limit: Int): List<Experience> {
        val terms = tokenize(query)
        return read(experiences, ListSerializer(Experience.serializer())).sortedByDescending { score(terms, it.task + " " + it.plan + " " + it.result, it.score, it.success) }.take(limit)
    }

    @Synchronized
    override suspend fun store(experience: Experience) {
        val current = read(experiences, ListSerializer(Experience.serializer()))
        val value = if (experience.id.isBlank()) experience.copy(id = stableId(experience.task, experience.createdAtEpochMs)) else experience
        experiences.writeText(json.encodeToString(ListSerializer(Experience.serializer()), current.filterNot { it.id == value.id } + value))
    }

    override suspend fun count(): Long = read(experiences, ListSerializer(Experience.serializer())).size.toLong()

    @Synchronized
    override suspend fun saveFact(fact: MemoryFact) {
        val current = read(facts, ListSerializer(MemoryFact.serializer()))
        facts.writeText(json.encodeToString(ListSerializer(MemoryFact.serializer()), current.filterNot { it.key == fact.key } + fact))
    }

    override suspend fun searchFacts(query: String, limit: Int): List<MemoryFact> {
        val terms = tokenize(query)
        return read(facts, ListSerializer(MemoryFact.serializer())).sortedByDescending {
            tokenize(it.key + " " + it.value).intersect(terms).size + it.confidence
        }.take(limit)
    }

    @Synchronized
    override suspend fun saveSkill(skill: Skill) {
        val current = read(skills, ListSerializer(Skill.serializer()))
        skills.writeText(json.encodeToString(ListSerializer(Skill.serializer()), current.filterNot { it.name == skill.name } + skill))
    }

    override suspend fun searchSkills(query: String, limit: Int): List<Skill> {
        val terms = tokenize(query)
        return read(skills, ListSerializer(Skill.serializer())).sortedByDescending {
            tokenize(it.name + " " + it.description + " " + it.triggerTerms.joinToString(" ")).intersect(terms).size + it.successRate
        }.take(limit)
    }

    private fun <T> read(file: File, serializer: KSerializer<List<T>>): List<T> =
        runCatching { if (!file.isFile) emptyList() else json.decodeFromString(serializer, file.readText()) }.getOrDefault(emptyList())

    private fun tokenize(s: String): Set<String> = s.lowercase().split(Regex("[^\\p{L}\\p{N}]+" )).filter { it.length > 2 }.toSet()
    private fun score(terms: Set<String>, text: String, base: Double, success: Boolean): Double =
        tokenize(text).intersect(terms).size * 1.0 + base * 0.25 + if (success) 0.2 else 0.0
    private fun stableId(seed: String, salt: Long): String = (seed.hashCode().toLong() xor salt).toString(16)
}
