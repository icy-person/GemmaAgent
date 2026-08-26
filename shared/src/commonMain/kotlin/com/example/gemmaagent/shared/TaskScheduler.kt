package com.example.gemmaagent.shared

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable

@Serializable
data class ScheduledTask(
    val id: String,
    val prompt: String,
    val intervalMs: Long,
    val enabled: Boolean = true,
)

class AgentTaskScheduler(
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default),
) {
    private val jobs = mutableMapOf<String, Job>()

    fun schedule(task: ScheduledTask, action: suspend (String) -> Unit) {
        cancel(task.id)
        if (!task.enabled) return
        jobs[task.id] = scope.launch {
            while (true) {
                action(task.prompt)
                delay(task.intervalMs.coerceAtLeast(1_000))
            }
        }
    }

    fun cancel(id: String) {
        jobs.remove(id)?.cancel()
    }

    fun cancelAll() {
        jobs.values.forEach(Job::cancel)
        jobs.clear()
    }

    fun activeCount(): Int = jobs.count { it.value.isActive }
}
