package com.example.gemmaagent.shared

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

val JsonElement.jsonObject: JsonObject get() = this as JsonObject
val JsonElement.jsonPrimitive: JsonPrimitive get() = this as JsonPrimitive
