package com.example.gemmaagent.shared

import kotlinx.atomicfu.atomic

class AgentMetrics : AgentObserver {
    private val thinking = atomic(0)
    private val toolCalls = atomic(0)
    private val toolFailures = atomic(0)
    private val completed = atomic(0)
    private val failed = atomic(0)

    override fun onEvent(event: AgentEvent) {
        when (event) {
            is AgentEvent.Thinking -> thinking.incrementAndGet()
            is AgentEvent.ToolRequested -> toolCalls.incrementAndGet()
            is AgentEvent.ToolCompleted -> if (!event.result.ok) toolFailures.incrementAndGet()
            is AgentEvent.Finished -> completed.incrementAndGet()
            is AgentEvent.Failed -> failed.incrementAndGet()
            is AgentEvent.Started, is AgentEvent.Reflection -> Unit
        }
    }

    fun snapshot(): Map<String, Long> = mapOf(
        "thinking" to thinking.value,
        "toolCalls" to toolCalls.value,
        "toolFailures" to toolFailures.value,
        "completed" to completed.value,
        "failed" to failed.value,
    )
}
