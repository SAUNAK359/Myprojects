package com.saunak.director.storage

import android.graphics.Bitmap
import android.net.Uri
import com.saunak.director.director.FinalRenderAdjustment
import com.saunak.director.director.PhotographicDirection
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.render.AnipEngine
import com.saunak.director.engine.render.AnipRenderResult
import com.saunak.director.engine.render.AnipRequest
import com.saunak.director.engine.render.AnipSituation
import com.saunak.director.engine.render.AnipSubjectContext
import com.saunak.director.processing.LookSpec
import com.saunak.director.processing.PhotoQuality
import com.saunak.director.processing.PhotographicDirectionFinisher
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

object AnipPhotoFinish {
    data class Output(val bitmap: Bitmap, val result: AnipRenderResult)

    fun render(
        source: Bitmap,
        look: LookSpec,
        sceneLabel: String,
        objective: String,
        reconstructionIntegrity: Double,
        direction: PhotographicDirection? = null,
        adjustment: FinalRenderAdjustment? = null
    ): Output {
        val engine = SierraEngine.getOrNull()
        val track = engine?.subjects?.latest()
        val subject = track?.let {
            val mask = it.mask
            AnipSubjectContext(
                bounds = it.bounds,
                maskWidth = mask?.width ?: 0,
                maskHeight = mask?.height ?: 0,
                mask = mask?.probabilities,
                maskConfidence = mask?.confidence ?: 0.0,
                depthSeparation = it.depth?.foregroundSeparation ?: 0.0,
                trackConfidence = it.confidence
            )
        }
        val resolvedLook = direction
            ?.takeIf { !it.abstain && it.themeConfidence >= 0.20 && it.selectedThemeId.isNotBlank() }
            ?.let { d -> engine?.cognition?.themeLook(d.selectedThemeId)?.let { themeLook -> LookSpec(themeLook, maxOf(look.normalizedStrength, d.themeStrength.toFloat())) } }
            ?: look
        val motionRisk = engine?.motion?.latest()?.conservativeRisk ?: 0.45
        val semanticConfidence = direction
            ?.takeUnless { it.abstain }
            ?.themeConfidence
            ?.coerceIn(0.0, 1.0)
            ?: 0.0
        val request = AnipRequest(
            sceneLabel = sceneLabel,
            objective = objective,
            look = resolvedLook.look,
            lookStrength = resolvedLook.normalizedStrength,
            subjectPresent = subject != null,
            motionRisk = motionRisk,
            reconstructionIntegrity = reconstructionIntegrity.coerceIn(0.0, 1.0),
            story = direction?.story.orEmpty(),
            mood = direction?.mood.orEmpty(),
            primarySubject = direction?.primarySubject.orEmpty(),
            narrativeRole = direction?.narrativeRole.orEmpty(),
            atmospherePriority = direction?.atmospherePriority?.coerceIn(0.0, 1.0) ?: 0.5,
            skinNaturalness = direction?.skinNaturalness?.coerceIn(0.0, 1.0) ?: 0.8,
            highlightProtectionIntent = direction?.highlightProtection?.coerceIn(0.0, 1.0) ?: 0.6,
            shadowDepthIntent = direction?.shadowDepth?.coerceIn(0.0, 1.0) ?: 0.5,
            midtoneContrastIntent = direction?.midtoneContrast?.coerceIn(0.0, 1.0) ?: 0.5,
            localContrastIntent = direction?.localContrast?.coerceIn(0.0, 1.0) ?: 0.5,
            colorIntensityIntent = direction?.colorIntensity?.coerceIn(0.0, 1.0) ?: 0.5,
            warmthBiasIntent = direction?.warmthBias?.coerceIn(-0.25, 0.25) ?: 0.0,
            subjectRelightIntentEv = direction?.subjectRelightEv?.coerceIn(-0.4, 0.8) ?: 0.0,
            backgroundSuppressionIntentEv = direction?.backgroundSuppressionEv?.coerceIn(-0.6, 0.1) ?: 0.0,
            directionConfidence = semanticConfidence
        )
        val (anip, result) = AnipEngine().render(source, request, subject)
        val finished = PhotographicDirectionFinisher.apply(anip, direction, adjustment, subject)
        val released = if (passesReleaseGate(source, finished, result)) finished else source

        // Recycle only intermediates not returned to the caller. Returning the untouched source is an
        // intentional fail-safe when every artistic candidate violates hard technical invariants.
        if (finished !== released && finished !== source && !finished.isRecycled) finished.recycle()
        if (anip !== finished && anip !== released && anip !== source && !anip.isRecycled) anip.recycle()
        return Output(released, result)
    }

    private fun passesReleaseGate(source: Bitmap, candidate: Bitmap, result: AnipRenderResult): Boolean {
        if (candidate.isRecycled) return false
        if (result.score.artifactPenalty >= 0.72 || result.score.total < 0.18) return false
        val before = runCatching { PhotoQuality.analyze(source) }.getOrNull() ?: return true
        val after = runCatching { PhotoQuality.analyze(candidate) }.getOrNull() ?: return false
        if (after.catastrophic) return false
        if (after.brightClipFraction > before.brightClipFraction + 0.025) return false
        if (after.darkClipFraction > before.darkClipFraction + 0.035) return false
        if (after.heuristicScore + 0.16 < before.heuristicScore) return false

        val darkSituation = result.recipe.situation in setOf(
            AnipSituation.NIGHT,
            AnipSituation.NIGHT_NEON,
            AnipSituation.STAGE,
            AnipSituation.SUNSET,
            AnipSituation.PORTRAIT_LOW_LIGHT,
            AnipSituation.SELFIE_LOW_LIGHT
        )
        val allowedBlackRise = if (darkSituation) 12.0 else 20.0
        if (before.shadowP05 < 45.0 && after.shadowP05 - before.shadowP05 > allowedBlackRise) return false

        val beforeSpan = (before.highlightP95 - before.shadowP05).coerceAtLeast(12.0)
        val afterSpan = (after.highlightP95 - after.shadowP05).coerceAtLeast(0.0)
        if (afterSpan / beforeSpan < if (darkSituation) 0.68 else 0.62) return false
        return true
    }
}

suspend fun PhotoRepository.saveAnipFinal(
    file: File,
    look: LookSpec,
    aspectRatio: String = "4:3",
    cropCenterX: Double = 0.5,
    cropCenterY: Double = 0.5,
    sceneLabel: String = "",
    objective: String = "",
    direction: PhotographicDirection? = null,
    adjustment: FinalRenderAdjustment? = null
): Uri {
    val source = decodeOriented(file, PhotoRepository.PROCESSED_MAX_DIMENSION)
    var rendered: Bitmap? = null
    var cropped: Bitmap? = null
    return try {
        val output = AnipPhotoFinish.render(
            source = source,
            look = look,
            sceneLabel = sceneLabel,
            objective = objective,
            reconstructionIntegrity = JPEG_EVIDENCE_INTEGRITY_PRIOR,
            direction = direction,
            adjustment = adjustment
        )
        rendered = output.bitmap
        val ratio = SierraCropPolicy.ratioOrNull(aspectRatio)
        cropped = if (ratio == null) output.bitmap else cropAnipBitmap(output.bitmap, ratio.first, ratio.second, cropCenterX, cropCenterY)
        saveProcessed(checkNotNull(cropped), file)
    } finally {
        cropped?.let { if (it !== rendered && !it.isRecycled) it.recycle() }
        rendered?.let { if (!it.isRecycled) it.recycle() }
        if (!source.isRecycled) source.recycle()
    }
}

internal fun cropAnipBitmap(
    source: Bitmap,
    ratioWidth: Int,
    ratioHeight: Int,
    centerX: Double,
    centerY: Double
): Bitmap {
    val target = ratioWidth.toDouble() / ratioHeight.toDouble()
    val current = source.width.toDouble() / source.height.toDouble()
    if (abs(target - current) < 0.015) return source
    val cropWidth: Int
    val cropHeight: Int
    if (current > target) {
        cropHeight = source.height
        cropWidth = (cropHeight * target).roundToInt().coerceIn(1, source.width)
    } else {
        cropWidth = source.width
        cropHeight = (cropWidth / target).roundToInt().coerceIn(1, source.height)
    }
    val safeCx = centerX.coerceIn(0.0, 1.0)
    val safeCy = centerY.coerceIn(0.0, 1.0)
    val left = (safeCx * source.width - cropWidth / 2.0).roundToInt().coerceIn(0, source.width - cropWidth)
    val top = (safeCy * source.height - cropHeight / 2.0).roundToInt().coerceIn(0, source.height - cropHeight)
    return Bitmap.createBitmap(source, left, top, cropWidth, cropHeight)
}

private const val JPEG_EVIDENCE_INTEGRITY_PRIOR = 0.78
