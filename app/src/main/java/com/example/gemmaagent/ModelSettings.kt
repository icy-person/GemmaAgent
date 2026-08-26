package com.example.gemmaagent

import android.content.Context
import com.example.gemmaagent.shared.AgentConfig
import com.example.gemmaagent.shared.AgentMode

 data class ModelSettings(
    val temperature: Float = 0.7f,
    val topK: Int = 64,
    val topP: Float = 0.95f,
    val maxIterations: Int = 50,
    val maxContextChars: Int = 100_000,
    val memoryTopK: Int = 8,
    val skillTopK: Int = 5,
    val reflectionEnabled: Boolean = true,
    val learnFromFailures: Boolean = true,
    val mode: AgentMode = AgentMode.ASSISTED,
) {
    fun agentConfig() = AgentConfig(
        maxIterations = maxIterations,
        maxContextChars = maxContextChars,
        memoryTopK = memoryTopK,
        skillTopK = skillTopK,
        reflectionEnabled = reflectionEnabled,
        learnFromFailures = learnFromFailures,
        mode = mode,
    )

    companion object {
        private const val PREFS = "agent_settings"
        fun load(context: Context): ModelSettings {
            val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            return ModelSettings(
                temperature = p.getFloat("temperature", 0.7f),
                topK = p.getInt("topK", 64),
                topP = p.getFloat("topP", 0.95f),
                maxIterations = p.getInt("maxIterations", 50),
                maxContextChars = p.getInt("maxContextChars", 100_000),
                memoryTopK = p.getInt("memoryTopK", 8),
                skillTopK = p.getInt("skillTopK", 5),
                reflectionEnabled = p.getBoolean("reflectionEnabled", true),
                learnFromFailures = p.getBoolean("learnFromFailures", true),
                mode = runCatching { AgentMode.valueOf(p.getString("mode", AgentMode.ASSISTED.name)!!) }.getOrDefault(AgentMode.ASSISTED),
            )
        }
    }

    fun save(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putFloat("temperature", temperature)
            .putInt("topK", topK)
            .putFloat("topP", topP)
            .putInt("maxIterations", maxIterations)
            .putInt("maxContextChars", maxContextChars)
            .putInt("memoryTopK", memoryTopK)
            .putInt("skillTopK", skillTopK)
            .putBoolean("reflectionEnabled", reflectionEnabled)
            .putBoolean("learnFromFailures", learnFromFailures)
            .putString("mode", mode.name)
            .apply()
    }
}
