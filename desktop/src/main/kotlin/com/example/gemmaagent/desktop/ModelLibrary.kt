package com.example.gemmaagent.desktop

import java.io.File
import java.util.prefs.Preferences

class ModelLibrary {
    private val prefs = Preferences.userRoot().node("com.example.gemmaagent/models")
    fun lastPath(): String = prefs.get("lastPath", "")
    fun remember(path: String) { if (path.isNotBlank()) { prefs.put("lastPath", path); prefs.putLong("lastUsed", System.currentTimeMillis()) } }
    fun validate(path: String): String? {
        val file = File(path)
        if (!file.exists()) return "Model file does not exist"
        if (!file.isFile) return "Model path is not a file"
        if (!file.canRead()) return "Model file is not readable"
        if (!path.endsWith(".litertlm", ignoreCase = true)) return "Expected a .litertlm model"
        return null
    }
    fun sizeBytes(path: String): Long = File(path).takeIf { it.isFile }?.length() ?: 0L
}

data class ModelBenchmark(val firstTokenMs: Long, val totalMs: Long, val estimatedTokens: Int, val tokensPerSecond: Double)
