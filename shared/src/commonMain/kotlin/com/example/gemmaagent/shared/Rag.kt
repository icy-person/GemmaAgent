package com.example.gemmaagent.shared

import kotlinx.serialization.Serializable

@Serializable
data class RagDocument(
    val id: String,
    val source: String,
    val title: String = "",
    val text: String,
    val metadata: Map<String, String> = emptyMap(),
)

@Serializable
data class RagChunk(
    val id: String,
    val documentId: String,
    val source: String,
    val title: String,
    val text: String,
    val start: Int,
    val end: Int,
)

data class RagHit(val chunk: RagChunk, val score: Double)

interface RagStore {
    suspend fun upsert(document: RagDocument)
    suspend fun delete(documentId: String)
    suspend fun clear()
    suspend fun countDocuments(): Long
    suspend fun countChunks(): Long
    suspend fun search(query: String, limit: Int = 8): List<RagHit>
}

class RagEngine(
    private val store: RagStore,
    private val chunkChars: Int = 1200,
    private val overlapChars: Int = 180,
) {
    suspend fun ingest(document: RagDocument) {
        require(document.id.isNotBlank())
        require(document.text.isNotBlank())
        store.upsert(document.copy(text = document.text.replace("\r\n", "\n")))
    }

    suspend fun retrieve(query: String, limit: Int = 8): List<RagHit> =
        store.search(query, limit.coerceIn(1, 32))

    suspend fun context(query: String, limit: Int = 6, maxChars: Int = 24000): String {
        val hits = retrieve(query, limit)
        return buildString {
            hits.forEachIndexed { index, hit ->
                append("[${index + 1}] ${hit.chunk.title.ifBlank { hit.chunk.source }}\\n")
                append("SOURCE: ${hit.chunk.source}\\n")
                append("SCORE: ${"%.3f".format(hit.score)}\\n")
                append(hit.chunk.text)
                append("\\n\\n")
            }
        }.take(maxChars)
    }

    fun chunk(document: RagDocument): List<RagChunk> {
        val text = document.text.trim()
        if (text.isEmpty()) return emptyList()
        val result = mutableListOf<RagChunk>()
        var start = 0
        var index = 0
        while (start < text.length) {
            val end = minOf(text.length, start + chunkChars)
            var actualEnd = end
            if (end < text.length) {
                val boundary = text.lastIndexOfAny(charArrayOf('\n', '.', '!', '?', ' '), endIndex = end - 1)
                if (boundary > start + chunkChars / 2) actualEnd = boundary
            }
            val chunkText = text.substring(start, actualEnd).trim()
            if (chunkText.isNotBlank()) result += RagChunk(
                id = "${document.id}:$index",
                documentId = document.id,
                source = document.source,
                title = document.title,
                text = chunkText,
                start = start,
                end = actualEnd,
            )
            if (actualEnd >= text.length) break
            start = maxOf(actualEnd - overlapChars, start + 1)
            index++
        }
        return result
    }
}

internal fun ragTokens(value: String): List<String> =
    value.lowercase().split(Regex("[^\\p{L}\\p{N}]+"))
        .filter { it.length >= 2 }
        .distinct()
