package com.saunak.director.ai

import com.saunak.director.director.EvaluationOutput
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import com.saunak.director.director.PlanOutput
import com.saunak.director.director.SierraAtmosphere
import com.saunak.director.director.SierraColorScience
import com.saunak.director.director.SierraFilmResponse
import com.saunak.director.director.SierraFinishing
import com.saunak.director.director.SierraHighlightBehavior
import com.saunak.director.director.SierraLocalLight
import com.saunak.director.director.SierraMaterialRendering
import com.saunak.director.director.SierraOpticalCharacter
import com.saunak.director.director.SierraRenderExecution
import com.saunak.director.director.SierraRenderSpec
import com.saunak.director.director.SierraShadowBehavior
import com.saunak.director.director.SierraSpatialMasks
import com.saunak.director.director.SierraSubjectHierarchy
import com.saunak.director.director.SierraTextureModel
import com.saunak.director.director.SierraTonalResponse
import com.saunak.director.director.ThemeBlendComponent
import com.saunak.director.tools.ToolIds

object VlmContractValidator {
    fun plan(value: PlanOutput): PlanOutput {
        val scene = clean(value.scene)
        val objective = clean(value.objective)
        val branch = clean(value.skillBranch)
        require(scene.isNotBlank()) { "Invalid scene" }
        require(objective.isNotBlank()) { "Invalid objective" }
        require(branch.isNotBlank()) { "Invalid branch" }
        require(value.referenceScore.isFinite()) { "Invalid reference score" }
        val calls = value.toolCalls.take(8)
        val cameraCalls = calls.filter { it.id in ToolIds.CAMERA_IDS }
        require(cameraCalls.map { it.id }.distinct().size == cameraCalls.size) { "Sierra plan is ambiguous: duplicate camera control" }
        require(calls.count { it.id in ToolIds.LOOK_IDS } <= 1) { "Sierra plan is ambiguous: multiple final looks" }
        val extension = calls.firstOrNull { it.id == ToolIds.EXTENSION }
        if (extension != null && extension.stringValue?.lowercase() != "none") {
            require(cameraCalls.none { it.id != ToolIds.EXTENSION }) { "Sierra plan is ambiguous: OEM extension and direct camera controls requested together" }
        }
        val flashOn = calls.any { it.id == ToolIds.FLASH && it.stringValue?.lowercase() != "off" }
        val torchOn = calls.any { it.id == ToolIds.TORCH && (it.value ?: 0.0) >= 0.5 }
        require(!(flashOn && torchOn)) { "Sierra plan is ambiguous: flash and torch requested together" }
        return value.copy(
            scene = scene.take(64), objective = objective.take(180), skillBranch = branch.take(48),
            referenceScore = value.referenceScore.coerceIn(0.0, 1.0), toolCalls = calls,
            cropCenterX = value.cropCenterX.coerceIn(0.0, 1.0), cropCenterY = value.cropCenterY.coerceIn(0.0, 1.0),
            photographicDirection = sanitizeDirection(value.photographicDirection)
        )
    }

    fun evaluation(value: EvaluationOutput): EvaluationOutput {
        val scores = listOf(value.score, value.technical, value.composition, value.subject, value.aesthetic, value.intentMatch, value.naturalness)
        require(scores.all { it.isFinite() }) { "Invalid evaluation score" }
        require(value.deltaCalls.isEmpty()) { "Post-capture Sierra render call must not request camera changes" }
        return value.copy(
            score = value.score.coerceIn(0.0, 1.0), technical = value.technical.coerceIn(0.0, 1.0),
            composition = value.composition.coerceIn(0.0, 1.0), subject = value.subject.coerceIn(0.0, 1.0),
            aesthetic = value.aesthetic.coerceIn(0.0, 1.0), intentMatch = value.intentMatch.coerceIn(0.0, 1.0),
            naturalness = value.naturalness.coerceIn(0.0, 1.0), issues = value.issues.take(8).map { clean(it).take(96) },
            deltaCalls = emptyList(), renderAdjustment = sanitizeAdjustment(value.renderAdjustment),
            renderSpec = value.renderSpec?.let(::renderSpec)
        )
    }

    fun renderSpec(v: SierraRenderSpec): SierraRenderSpec {
        require(v.schemaVersion == 1) { "Unsupported Sierra render schema" }
        val precision = clean(v.execution.precision)
        require(precision == "linear_srgb_argb8888") { "Unsupported Sierra render precision" }
        return v.copy(
            intentSummary = clean(v.intentSummary).take(180),
            tonal = SierraTonalResponse(
                exposureEv = finite(v.tonal.exposureEv).coerceIn(-2.0, 2.0),
                blackPoint = finite(v.tonal.blackPoint).coerceIn(-0.25, 0.35),
                toeStrength = finite(v.tonal.toeStrength).coerceIn(0.0, 1.0),
                midtoneContrast = finite(v.tonal.midtoneContrast).coerceIn(-1.0, 1.0),
                shoulderStrength = finite(v.tonal.shoulderStrength).coerceIn(0.0, 1.0),
                dynamicRangeCompression = finite(v.tonal.dynamicRangeCompression).coerceIn(0.0, 1.0)
            ),
            color = SierraColorScience(
                temperature = finite(v.color.temperature).coerceIn(-1.0, 1.0), tint = finite(v.color.tint).coerceIn(-1.0, 1.0),
                saturation = finite(v.color.saturation).coerceIn(0.0, 1.8), vibrance = finite(v.color.vibrance).coerceIn(-1.0, 1.0),
                shadowHueShiftDeg = finite(v.color.shadowHueShiftDeg).coerceIn(-60.0, 60.0), shadowColorStrength = finite(v.color.shadowColorStrength).coerceIn(0.0, 1.0),
                highlightHueShiftDeg = finite(v.color.highlightHueShiftDeg).coerceIn(-60.0, 60.0), highlightColorStrength = finite(v.color.highlightColorStrength).coerceIn(0.0, 1.0),
                skinProtection = finite(v.color.skinProtection).coerceIn(0.0, 1.0)
            ),
            localLight = SierraLocalLight(
                subjectEv = finite(v.localLight.subjectEv).coerceIn(-1.5, 1.5), backgroundEv = finite(v.localLight.backgroundEv).coerceIn(-1.5, 1.0),
                faceEv = finite(v.localLight.faceEv).coerceIn(-0.75, 0.75), rimEmphasis = finite(v.localLight.rimEmphasis).coerceIn(0.0, 1.0),
                dodgeBurnStrength = finite(v.localLight.dodgeBurnStrength).coerceIn(0.0, 1.0)
            ),
            hierarchy = SierraSubjectHierarchy(
                subjectSeparation = finite(v.hierarchy.subjectSeparation).coerceIn(0.0, 1.0), backgroundPresence = finite(v.hierarchy.backgroundPresence).coerceIn(0.0, 1.0),
                luminousSourcePriority = finite(v.hierarchy.luminousSourcePriority).coerceIn(0.0, 1.0), textPreservation = finite(v.hierarchy.textPreservation).coerceIn(0.0, 1.0)
            ),
            materials = SierraMaterialRendering(
                skinTexture = finite(v.materials.skinTexture).coerceIn(0.0, 1.0), hairDetail = finite(v.materials.hairDetail).coerceIn(0.0, 1.0),
                fabricDetail = finite(v.materials.fabricDetail).coerceIn(0.0, 1.0), foliageDetail = finite(v.materials.foliageDetail).coerceIn(0.0, 1.0),
                metalSpecular = finite(v.materials.metalSpecular).coerceIn(0.0, 1.0), glassSpecular = finite(v.materials.glassSpecular).coerceIn(0.0, 1.0)
            ),
            highlights = SierraHighlightBehavior(
                rolloff = finite(v.highlights.rolloff).coerceIn(0.0, 1.0), compression = finite(v.highlights.compression).coerceIn(0.0, 1.0),
                bloom = finite(v.highlights.bloom).coerceIn(0.0, 1.0), halation = finite(v.highlights.halation).coerceIn(0.0, 1.0),
                practicalLightPreservation = finite(v.highlights.practicalLightPreservation).coerceIn(0.0, 1.0)
            ),
            shadows = SierraShadowBehavior(
                depth = finite(v.shadows.depth).coerceIn(0.0, 1.0), recovery = finite(v.shadows.recovery).coerceIn(0.0, 1.0),
                blackAnchor = finite(v.shadows.blackAnchor).coerceIn(0.0, 1.0), colorStrength = finite(v.shadows.colorStrength).coerceIn(0.0, 1.0)
            ),
            texture = SierraTextureModel(
                denoise = finite(v.texture.denoise).coerceIn(0.0, 1.0), chromaDenoise = finite(v.texture.chromaDenoise).coerceIn(0.0, 1.0),
                detail = finite(v.texture.detail).coerceIn(0.0, 1.0), microcontrast = finite(v.texture.microcontrast).coerceIn(0.0, 1.0),
                sharpening = finite(v.texture.sharpening).coerceIn(0.0, 1.0), deblur = finite(v.texture.deblur).coerceIn(0.0, 1.0)
            ),
            optical = SierraOpticalCharacter(
                diffusion = finite(v.optical.diffusion).coerceIn(0.0, 1.0), vignette = finite(v.optical.vignette).coerceIn(0.0, 1.0),
                edgeSoftness = finite(v.optical.edgeSoftness).coerceIn(0.0, 1.0), lensFalloff = finite(v.optical.lensFalloff).coerceIn(0.0, 1.0)
            ),
            film = SierraFilmResponse(
                curveStrength = finite(v.film.curveStrength).coerceIn(0.0, 1.0), grainAmount = finite(v.film.grainAmount).coerceIn(0.0, 0.5),
                grainSize = finite(v.film.grainSize).coerceIn(0.0, 1.0), colorResponse = finite(v.film.colorResponse).coerceIn(-1.0, 1.0),
                character = clean(v.film.character).take(48)
            ),
            atmosphere = SierraAtmosphere(
                dehaze = finite(v.atmosphere.dehaze).coerceIn(0.0, 1.0), glow = finite(v.atmosphere.glow).coerceIn(0.0, 1.0),
                haze = finite(v.atmosphere.haze).coerceIn(0.0, 1.0), localContrast = finite(v.atmosphere.localContrast).coerceIn(-1.0, 1.0)
            ),
            spatial = SierraSpatialMasks(
                useSubjectMask = v.spatial.useSubjectMask, useDepth = v.spatial.useDepth,
                useFaceMask = false, useSkinMask = false, useSkyMask = false,
                feather = finite(v.spatial.feather).coerceIn(0.0, 1.0)
            ),
            finishing = SierraFinishing(
                outputContrast = finite(v.finishing.outputContrast).coerceIn(-1.0, 1.0),
                outputSaturation = finite(v.finishing.outputSaturation).coerceIn(0.0, 1.8),
                outputSharpen = finite(v.finishing.outputSharpen).coerceIn(0.0, 1.0), preserveBlacks = v.finishing.preserveBlacks
            ),
            execution = SierraRenderExecution(
                workingLongEdge = v.execution.workingLongEdge.coerceIn(1024, 4096), precision = precision,
                subjectMaskMethod = clean(v.execution.subjectMaskMethod).take(64), depthMethod = clean(v.execution.depthMethod).take(64),
                operations = v.execution.operations.take(32).map { clean(it).take(48) }, deviceRationale = clean(v.execution.deviceRationale).take(180)
            ),
            reasonCodes = v.reasonCodes.take(8).map { clean(it).take(64) }
        )
    }

    private fun sanitizeDirection(v: PhotographicDirection): PhotographicDirection {
        val blend = v.themeBlend.take(3).mapNotNull { component ->
            val id = clean(component.themeId).take(64)
            if (id.isBlank() || !component.weight.isFinite()) null else ThemeBlendComponent(id, component.weight.coerceIn(0.0, 1.0))
        }
        return v.copy(
            story = clean(v.story).take(160), mood = clean(v.mood).take(96), primarySubject = clean(v.primarySubject).take(96),
            narrativeRole = clean(v.narrativeRole).take(48), selectedThemeId = clean(v.selectedThemeId).take(64),
            themeStrength = finite(v.themeStrength).coerceIn(0.0, 1.0), themeConfidence = finite(v.themeConfidence).coerceIn(0.0, 1.0), themeBlend = blend,
            atmospherePriority = finite(v.atmospherePriority).coerceIn(0.0, 1.0), skinNaturalness = finite(v.skinNaturalness).coerceIn(0.0, 1.0),
            highlightProtection = finite(v.highlightProtection).coerceIn(0.0, 1.0), shadowDepth = finite(v.shadowDepth).coerceIn(0.0, 1.0),
            midtoneContrast = finite(v.midtoneContrast).coerceIn(0.0, 1.0), localContrast = finite(v.localContrast).coerceIn(0.0, 1.0),
            colorIntensity = finite(v.colorIntensity).coerceIn(0.0, 1.0), warmthBias = finite(v.warmthBias).coerceIn(-0.25, 0.25),
            subjectRelightEv = finite(v.subjectRelightEv).coerceIn(-0.4, 0.8), backgroundSuppressionEv = finite(v.backgroundSuppressionEv).coerceIn(-0.6, 0.1),
            grain = finite(v.grain).coerceIn(0.0, 0.20), vignette = finite(v.vignette).coerceIn(0.0, 0.35),
            reasonCodes = v.reasonCodes.take(8).map { clean(it).take(64) }
        )
    }

    private fun sanitizeAdjustment(v: FinalRenderAdjustment): FinalRenderAdjustment = v.copy(
        themeStrengthDelta = finite(v.themeStrengthDelta).coerceIn(-0.20, 0.20), midtoneContrastDelta = finite(v.midtoneContrastDelta).coerceIn(-0.25, 0.25),
        localContrastDelta = finite(v.localContrastDelta).coerceIn(-0.25, 0.25), shadowDepthDelta = finite(v.shadowDepthDelta).coerceIn(-0.20, 0.20),
        highlightProtectionDelta = finite(v.highlightProtectionDelta).coerceIn(-0.20, 0.20), subjectRelightDeltaEv = finite(v.subjectRelightDeltaEv).coerceIn(-0.25, 0.25),
        backgroundSuppressionDeltaEv = finite(v.backgroundSuppressionDeltaEv).coerceIn(-0.20, 0.20), warmthDelta = finite(v.warmthDelta).coerceIn(-0.10, 0.10),
        colorIntensityDelta = finite(v.colorIntensityDelta).coerceIn(-0.15, 0.15)
    )

    private fun finite(v: Double): Double = if (v.isFinite()) v else 0.0
    private fun clean(value: String): String = value.filter { it >= ' ' && it != '\u007f' }.trim()
}
