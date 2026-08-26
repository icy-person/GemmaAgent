package com.example.gemmaagent.shared

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import kotlin.math.log

@Serializable
private data class RagState(val documents: List<RagDocument> = emptyList())

class JvmRagStore(private val file: File) : RagStore {
    private val mutex = Mutex()
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val docs = linkedMapOf<String, RagDocument>()
    private val engine = RagEngine(this)
    private var loaded = false

    private suspend fun ensureLoaded() = mutex.withLock {
        if (loaded) return@withLock
        if (file.isFile) runCatching {
            val state = json.decodeFromString<RagState>(file.readText(StandardCharsets.UTF_8))
            state.documents.forEach { docs[it.id] = it }
        }
        loaded = true
    }

    override suspend fun upsert(document: RagDocument) {
        ensureLoaded()
        mutex.withLock { docs[document.id] = document; saveLocked() }
    }

    override suspend fun delete(documentId: String) {
        ensureLoaded()
        mutex.withLock { docs.remove(documentId); saveLocked() }
    }

    override suspend fun clear() {
        ensureLoaded()
        mutex.withLock { docs.clear(); saveLocked() }
    }

    override suspend fun countDocuments(): Long {
        ensureLoaded()
        return mutex.withLock { docs.size.toLong() }
    }

    override suspend fun countChunks(): Long {
        ensureLoaded()
        return mutex.withLock { docs.values.sumOf { engine.chunk(it).size }.toLong() }
    }

    override suspend fun search(query: String, limit: Int): List<RagHit> {
        ensureLoaded()
        val tokens = ragTokens(query)
        if (tokens.isEmpty()) return emptyList()
        return mutex.withLock {
            val chunks = docs.values.flatMap { engine.chunk(it) }
            if (chunks.isEmpty()) return@withLock emptyList()
            val documentFrequency = tokens.associateWith { token ->
                chunks.count { ragTokens(it.text).contains(token) }.coerceAtLeast(1)
            }
            chunks.mapNotNull { chunk ->
                val terms = ragTokens(chunk.text).toSet()
                val matched = tokens.count { it in terms }
                if (matched == 0) return@mapNotNull null
                val tf = matched.toDouble() / tokens.size
                val idf = tokens.sumOf { token ->
                    log((chunks.size + 1.0) / documentFrequency.getValue(token))
                } / tokens.size
                val titleBoost = if (tokens.any { chunk.title.lowercase().contains(it) }) 0.15 else 0.0
                RagHit(chunk, (tf * 0.75 + idf * 0.12 + titleBoost).coerceIn(0.0, 1.0 + titleBoost))
            }.sortedByDescending { it.score }.take(limit.coerceIn(1, 32))
        }
    }

    private fun saveLocked() {
        val target = file.toPath().toAbsolutePath().normalize()
        val parent = target.parent ?: Path.of(".").toAbsolutePath().normalize()
        Files.createDirectories(parent)
        val temp = parent.resolve(target.fileName.toString() + ".part")
        val payload = json.encodeToString(RagState(docs.values.toList()))
        Files.writeString(temp, payload, StandardCharsets.UTF_8)
        runCatching {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
        }.getOrElse {
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING)
        }
    }
}
