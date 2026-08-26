package com.example.gemmaagent.shared

data class ModelBenchmark(
    val firstTokenMs: Long,
    val totalMs: Long,
    val estimatedTokens: Int,
    val tokensPerSecond: Double,
)
