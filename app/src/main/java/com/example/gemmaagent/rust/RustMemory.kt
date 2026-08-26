package com.example.gemmaagent.rust

import com.example.gemmaagent.shared.Experience
import kotlinx.serialization.json.Json

class RustMemory(private val path: String) : AutoCloseable {
    private var handle: Long = nativeCreate(path)
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun search(query: String, limit: Int = 5): List<Experience> =
        if (handle == 0L) emptyList() else runCatching {
            json.decodeFromString<List<Experience>>(nativeSearch(handle, query, limit))
        }.getOrDefault(emptyList())

    fun store(experience: Experience): String = if (handle == 0L) "" else nativeStore(handle, json.encodeToString(experience))

    fun count(): Long = if (handle == 0L) 0 else nativeCount(handle)

    override fun close() {
        if (handle != 0L) { nativeDestroy(handle); handle = 0L }
    }

    private external fun nativeCreate(path: String): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeSearch(handle: Long, query: String, limit: Int): String
    private external fun nativeStore(handle: Long, experienceJson: String): String
    private external fun nativeCount(handle: Long): Long

    companion object {
        init { System.loadLibrary("gemma_agent_core") }
    }
}
