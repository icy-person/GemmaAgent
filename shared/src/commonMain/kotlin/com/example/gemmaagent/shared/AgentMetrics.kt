package com.example.gemmaagent.shared

class AgentMetrics : AgentObserver {
    private var thinking = 0L
    private var toolCalls = 0L
    private var toolFailures = 0L
    private var completed = 0L
    private var failed = 0L
    private var modelOutputs = 0L
    private var stages = 0L

    @Synchronized
    override fun onEvent(event: AgentEvent) {
        when (event) {
            is AgentEvent.Thinking -> thinking++
            is AgentEvent.ToolRequested -> toolCalls++
            is AgentEvent.ToolCompleted -> if (!event.result.ok) toolFailures++
            is AgentEvent.Finished -> completed++
            is AgentEvent.Failed -> failed++
            is AgentEvent.ModelOutput -> modelOutputs++
            is AgentEvent.Stage -> stages++
            is AgentEvent.Started, is AgentEvent.Reflection -> Unit
        }
    }

    @Synchronized
    fun snapshot(): Map<String, Long> = mapOf(
        "thinking" to thinking,
        "toolCalls" to toolCalls,
        "toolFailures" to toolFailures,
        "modelOutputs" to modelOutputs,
        "stages" to stages,
        "completed" to completed,
        "failed" to failed,
    )
}
