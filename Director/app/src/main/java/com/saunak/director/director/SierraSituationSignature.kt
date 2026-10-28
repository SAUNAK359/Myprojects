package com.saunak.director.director

import com.saunak.director.processing.TechnicalQuality
import kotlinx.serialization.Serializable

@Serializable
enum class DirectorStrategy {
    PORTRAIT,
    CANDID,
    GROUP,
    FASHION,
    FOOD,
    PRODUCT,
    ARCHITECTURE,
    LANDSCAPE,
    NIGHT,
    GENERAL
}

@Serializable
enum class InterventionPolicy {
    OBSERVE,
    MINIMAL,
    DIRECTED,
    HIGH_DIRECTION
}

/**
 * Compact, measurement-backed description of the photographic situation. It is deliberately not a
 * single scene label: Gemini can reason over the independent dimensions while Sierra beta can make
 * deterministic local choices from the same evidence.
 */
data class SierraSituationSignature(
    val coarseScene: String,
    val primaryHuman: Boolean,
    val peopleCount: Int,
    val faceCoverage: Double,
    val lowLight: Boolean,
    val brightSky: Boolean,
    val structuredEnvironment: Boolean,
    val highDynamicRange: Boolean,
    val flatLight: Boolean,
    val subjectSeparation: Double,
    val depthConfidence: Double,
    val landscapeOrientation: Boolean,
    val portraitOrientation: Boolean
) {
    val recommendedStrategy: DirectorStrategy
        get() = when {
            lowLight && primaryHuman -> DirectorStrategy.NIGHT
            lowLight -> DirectorStrategy.NIGHT
            primaryHuman && peopleCount > 1 -> DirectorStrategy.GROUP
            primaryHuman -> DirectorStrategy.PORTRAIT
            coarseScene == "architecture" || structuredEnvironment -> DirectorStrategy.ARCHITECTURE
            coarseScene == "landscape" || brightSky -> DirectorStrategy.LANDSCAPE
            else -> DirectorStrategy.GENERAL
        }

    val recommendedIntervention: InterventionPolicy
        get() = when (recommendedStrategy) {
            DirectorStrategy.CANDID -> InterventionPolicy.MINIMAL
            DirectorStrategy.LANDSCAPE, DirectorStrategy.ARCHITECTURE -> InterventionPolicy.MINIMAL
            DirectorStrategy.GROUP, DirectorStrategy.FASHION, DirectorStrategy.PRODUCT -> InterventionPolicy.DIRECTED
            else -> InterventionPolicy.DIRECTED
        }

    fun compactDescription(): String = buildString {
        append("SITUATION_SIGNATURE{")
        append("coarse_scene=").append(coarseScene)
        append(";primary_human=").append(primaryHuman)
        append(";people=").append(peopleCount)
        append(";face_coverage=").append("%.3f".format(java.util.Locale.US, faceCoverage))
        append(";low_light=").append(lowLight)
        append(";bright_sky=").append(brightSky)
        append(";structured_environment=").append(structuredEnvironment)
        append(";high_dynamic_range=").append(highDynamicRange)
        append(";flat_light=").append(flatLight)
        append(";subject_separation=").append("%.3f".format(java.util.Locale.US, subjectSeparation))
        append(";depth_confidence=").append("%.3f".format(java.util.Locale.US, depthConfidence))
        append(";orientation=").append(if (portraitOrientation) "portrait" else if (landscapeOrientation) "landscape" else "balanced")
        append(";local_strategy_prior=").append(recommendedStrategy)
        append(";local_intervention_prior=").append(recommendedIntervention)
        append('}')
    }

    companion object {
        fun from(signals: SierraSceneSignals, quality: TechnicalQuality, width: Int, height: Int): SierraSituationSignature {
            require(width > 0 && height > 0)
            val low = quality.meanLuma < 72.0 && quality.medianLuma < 68.0
            val hdr = quality.highlightP95 - quality.shadowP05 > 145.0 || quality.lumaStdDev > 50.0 || signals.brightSky
            return SierraSituationSignature(
                coarseScene = signals.branch,
                primaryHuman = signals.primaryHumanSubject,
                peopleCount = signals.faceCount,
                faceCoverage = signals.faceCoverage.coerceIn(0.0, 1.0),
                lowLight = low,
                brightSky = signals.brightSky,
                structuredEnvironment = signals.urbanStructure,
                highDynamicRange = hdr,
                flatLight = quality.lumaStdDev < 24.0,
                subjectSeparation = signals.subjectSeparation.coerceIn(0.0, 1.0),
                depthConfidence = signals.relativeDepthConfidence.coerceIn(0.0, 1.0),
                landscapeOrientation = width > height * 1.12,
                portraitOrientation = height > width * 1.12
            )
        }
    }
}
