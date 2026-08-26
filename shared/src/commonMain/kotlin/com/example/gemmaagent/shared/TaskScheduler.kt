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
    private val lock = Any()
    private val jobs = mutableMapOf<String, Job>()

    fun schedule(task: ScheduledTask, action: suspend (String) -> Unit) {
        require(task.id.isNotBlank()) { "Task id is required" }
        require(task.prompt.isNotBlank()) { "Task prompt is required" }
        cancel(task.id)
        if (!task.enabled) return
        val job = scope.launch {
            while (true) {
                runCatching { action(task.prompt) }
                delay(task.intervalMs.coerceAtLeast(1_000))
            }
        }
        synchronized(lock) { jobs[task.id] = job }
        job.invokeOnCompletion { synchronized(lock) { if (jobs[task.id] === job) jobs.remove(task.id) } }
    }

    fun cancel(id: String) {
        synchronized(lock) { jobs.remove(id)?.cancel() }
    }

    fun cancelAll() {
        synchronized(lock) {
            jobs.values.forEach(Job::cancel)
            jobs.clear()
        }
    }

    fun activeCount(): Int = synchronized(lock) { jobs.values.count { it.isActive } }
}
