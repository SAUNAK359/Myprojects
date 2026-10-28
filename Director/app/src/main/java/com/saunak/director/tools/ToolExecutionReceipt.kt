package com.saunak.director.tools

import com.saunak.director.director.ToolCall

enum class ToolExecutionStatus { APPLIED, CLAMPED, FAILED, REJECTED }

/** Auditable requested -> resolved -> actual camera execution record. */
data class ToolExecutionReceipt(
    val requested: ToolCall,
    val resolved: ToolCall? = null,
    val actual: String? = null,
    val status: ToolExecutionStatus,
    val reason: String? = null,
    val timestampMs: Long = android.os.SystemClock.elapsedRealtime()
) {
    fun compactDescription(): String {
        val name = ToolNames.name(requested.id)
        val request = ToolNames.value(requested)
        val resolvedValue = resolved?.let(ToolNames::value)
        val suffix = when {
            status == ToolExecutionStatus.FAILED || status == ToolExecutionStatus.REJECTED -> reason?.let { " • $it" }.orEmpty()
            actual != null -> " → $actual"
            resolvedValue != null && resolvedValue != request -> " → $resolvedValue"
            else -> ""
        }
        return "${status.name.lowercase()} • $name $request$suffix"
    }
}

object ToolNames {
    fun name(id: Int): String = when (id) {
        ToolIds.FOCUS -> "Focus"
        ToolIds.ZOOM -> "Zoom"
        ToolIds.EXPOSURE_EV -> "Exposure"
        ToolIds.TORCH -> "Torch"
        ToolIds.FLASH -> "Flash"
        ToolIds.EXTENSION -> "OEM mode"
        ToolIds.COMPOSITION -> "Composition"
        ToolIds.LOOK_NATURAL -> "Natural"
        ToolIds.LOOK_WARM -> "Warm"
        ToolIds.LOOK_COOL -> "Cool"
        ToolIds.LOOK_CINEMATIC_WARM -> "Cinematic Warm"
        ToolIds.LOOK_CINEMATIC_COOL -> "Cinematic Cool"
        ToolIds.LOOK_MONOCHROME -> "Monochrome"
        ToolIds.LOOK_SOFT_PORTRAIT -> "Soft Portrait"
        ToolIds.LOOK_LANDSCAPE_VIVID -> "Landscape Vivid"
        ToolIds.LOOK_SOCIAL_POP -> "Social Pop"
        ToolIds.LOOK_FILM_TEAL_ORANGE -> "Film Teal/Orange"
        ToolIds.LOOK_GOLDEN_HOUR -> "Golden Hour"
        ToolIds.LOOK_MOODY -> "Moody"
        ToolIds.LOOK_CLEAN_PORTRAIT -> "Clean Portrait"
        ToolIds.LOOK_NIGHT_NEON -> "Night Neon"
        else -> "Tool $id"
    }

    fun value(call: ToolCall): String = when (call.id) {
        ToolIds.FOCUS -> "(${fmt(call.x)},${fmt(call.y)})"
        ToolIds.ZOOM -> "${fmt(call.value)}×"
        ToolIds.EXPOSURE_EV -> "${fmtSigned(call.value)} EV"
        ToolIds.TORCH -> if ((call.value ?: 0.0) >= .5) "on" else "off"
        ToolIds.FLASH, ToolIds.EXTENSION, ToolIds.COMPOSITION -> call.stringValue.orEmpty().ifBlank { "auto" }
        in ToolIds.LOOK_IDS -> call.value?.let { "${fmt(it)} strength" } ?: "default"
        else -> call.stringValue ?: call.value?.toString().orEmpty()
    }

    private fun fmt(value: Double?): String = value?.let { "%.2f".format(java.util.Locale.US, it) } ?: "?"
    private fun fmtSigned(value: Double?): String = value?.let { "%+.2f".format(java.util.Locale.US, it) } ?: "?"
}
