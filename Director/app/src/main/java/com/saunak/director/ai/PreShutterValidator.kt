package com.saunak.director.ai

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.director.PreShutterValidationOutput
import com.saunak.director.director.live.LiveDirectorValidator
import com.saunak.director.tools.ToolValidator

/** Fail-closed validator for Gemini call #2. */
object PreShutterValidator {
    fun validate(
        value: PreShutterValidationOutput,
        capabilities: CapabilityRegistry,
        directorContract: DirectorSchemaCompiler.Contract
    ): PreShutterValidationOutput {
        val scores = listOf(
            value.intentMatch, value.compositionMatch, value.subjectMatch,
            value.lightingMatch, value.backgroundMatch, value.cameraStateMatch
        )
        require(scores.all(Double::isFinite)) { "Invalid pre-shutter semantic score" }
        require(value.correctionCalls.size <= 8) { "Too many pre-shutter correction calls" }

        if (value.accept) {
            require(value.correctionCalls.isEmpty()) { "Accepted pre-shutter result cannot request camera corrections" }
            return value.copy(
                intentMatch = value.intentMatch.coerceIn(0.0, 1.0),
                compositionMatch = value.compositionMatch.coerceIn(0.0, 1.0),
                subjectMatch = value.subjectMatch.coerceIn(0.0, 1.0),
                lightingMatch = value.lightingMatch.coerceIn(0.0, 1.0),
                backgroundMatch = value.backgroundMatch.coerceIn(0.0, 1.0),
                cameraStateMatch = value.cameraStateMatch.coerceIn(0.0, 1.0),
                issues = value.issues.take(8).map { clean(it).take(120) },
                correctionCalls = emptyList(),
                correctedDirectorSpec = null,
                reason = clean(value.reason).take(180)
            )
        }

        val toolResult = ToolValidator().validate(value.correctionCalls, capabilities)
        require(toolResult.rejected.isEmpty()) {
            "Pre-shutter correction contains unsupported camera control: ${toolResult.rejected.firstOrNull()?.reason}"
        }
        val corrected = value.correctedDirectorSpec?.let { LiveDirectorValidator.validate(it, directorContract) }
        require(toolResult.accepted.isNotEmpty() || corrected != null) {
            "Rejected pre-shutter result must provide one executable correction"
        }
        return value.copy(
            intentMatch = value.intentMatch.coerceIn(0.0, 1.0),
            compositionMatch = value.compositionMatch.coerceIn(0.0, 1.0),
            subjectMatch = value.subjectMatch.coerceIn(0.0, 1.0),
            lightingMatch = value.lightingMatch.coerceIn(0.0, 1.0),
            backgroundMatch = value.backgroundMatch.coerceIn(0.0, 1.0),
            cameraStateMatch = value.cameraStateMatch.coerceIn(0.0, 1.0),
            issues = value.issues.take(8).map { clean(it).take(120) },
            correctionCalls = toolResult.accepted,
            correctedDirectorSpec = corrected,
            reason = clean(value.reason).take(180)
        )
    }

    private fun clean(value: String): String = value.filter { it >= ' ' && it != '\u007f' }.trim()
}
