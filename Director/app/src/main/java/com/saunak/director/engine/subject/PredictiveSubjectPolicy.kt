package com.saunak.director.engine.subject

import kotlin.math.exp

/** Pure, hardware-independent decision output derived from one temporal subject track. */
data class PredictiveSubjectDirective(
    val autofocusPoint: NormalizedPoint,
    val meteringPoint: NormalizedPoint,
    val cropCenter: NormalizedPoint,
    val edgeRisk: Double,
    val predictedDisplacement: Double,
    val kinematicRisk: Double,
    val confidence: Double,
    val exposureCapMultiplier: Double,
    val shouldRetargetThreeA: Boolean
) {
    init {
        require(edgeRisk in 0.0..1.0)
        require(predictedDisplacement >= 0.0)
        require(kinematicRisk in 0.0..1.0)
        require(confidence in 0.0..1.0)
        require(exposureCapMultiplier in 0.25..1.0)
    }
}

/**
 * Converts tracked geometry and kinematics into bounded photographic intent. This class never touches
 * CameraX; it can be unit-tested independently and ignored safely when track confidence is weak.
 */
object PredictiveSubjectPolicy {
    fun evaluate(track: SubjectTrack): PredictiveSubjectDirective {
        val predicted = track.predictedCenter.clamped()
        val current = track.center
        val displacement = current.distanceTo(predicted)
        val speed = track.kinematics.speedPerSec
        val acceleration = track.kinematics.accelerationPerSec2

        val focusBase = faceOrUpperBodyPoint(track)
        val dx = predicted.x - current.x
        val dy = predicted.y - current.y
        val autofocus = NormalizedPoint(focusBase.x + dx, focusBase.y + dy).clamped()

        val meterBase = track.mask?.centroid ?: current
        val metering = NormalizedPoint(meterBase.x + dx * 0.65, meterBase.y + dy * 0.65).clamped()

        val leadX = track.kinematics.velocityXPerSec.coerceIn(-1.0, 1.0) * LEAD_ROOM_GAIN
        val leadY = track.kinematics.velocityYPerSec.coerceIn(-0.6, 0.6) * VERTICAL_LEAD_GAIN
        val cropCenter = NormalizedPoint(predicted.x + leadX, predicted.y + leadY)
            .clampedToCompositionRange()

        val predictedBounds = translate(track.bounds, dx, dy)
        val edgeRisk = edgeRisk(predictedBounds)
        val speedRisk = (speed / SPEED_REFERENCE).coerceIn(0.0, 1.0)
        val accelRisk = (acceleration / ACCEL_REFERENCE).coerceIn(0.0, 1.0)
        val kinematicRisk = (0.58 * speedRisk + 0.24 * accelRisk + 0.18 * edgeRisk).coerceIn(0.0, 1.0)
        val missPenalty = Math.pow(MISS_CONFIDENCE_DECAY, track.missedUpdates.toDouble())
        val confidence = (track.confidence * (1.0 - 0.65 * track.uncertainty) * missPenalty).coerceIn(0.0, 1.0)
        val exposureMultiplier = exp(-EXPOSURE_GAIN * kinematicRisk)
            .coerceIn(MIN_EXPOSURE_MULTIPLIER, 1.0)
        val firstAcquisition = track.updateCount == 1 && track.missedUpdates == 0 &&
            confidence >= INITIAL_ACQUIRE_CONFIDENCE
        val predictiveRetarget = displacement >= RETARGET_DISPLACEMENT || speed >= RETARGET_SPEED || edgeRisk >= 0.45
        val retarget = track.missedUpdates == 0 && confidence >= RETARGET_CONFIDENCE &&
            (firstAcquisition || predictiveRetarget)

        return PredictiveSubjectDirective(
            autofocusPoint = autofocus,
            meteringPoint = metering,
            cropCenter = cropCenter,
            edgeRisk = edgeRisk,
            predictedDisplacement = displacement,
            kinematicRisk = kinematicRisk,
            confidence = confidence,
            exposureCapMultiplier = exposureMultiplier,
            shouldRetargetThreeA = retarget
        )
    }

    private fun faceOrUpperBodyPoint(track: SubjectTrack): NormalizedPoint {
        val facial = track.pose?.keypoints
            ?.filter { it.index in 0..4 && it.confidence >= 0.25 }
            .orEmpty()
        if (facial.isNotEmpty()) {
            val weight = facial.sumOf { it.confidence }.coerceAtLeast(1e-6)
            return NormalizedPoint(
                facial.sumOf { it.position.x * it.confidence } / weight,
                facial.sumOf { it.position.y * it.confidence } / weight
            ).clamped()
        }
        return NormalizedPoint(
            track.bounds.center.x,
            track.bounds.top + track.bounds.height * 0.22
        ).clamped()
    }

    private fun translate(rect: NormalizedRect, dx: Double, dy: Double): NormalizedRect = NormalizedRect(
        rect.left + dx,
        rect.top + dy,
        rect.right + dx,
        rect.bottom + dy
    )

    private fun edgeRisk(rect: NormalizedRect): Double {
        val leftMargin = rect.left
        val topMargin = rect.top
        val rightMargin = 1.0 - rect.right
        val bottomMargin = 1.0 - rect.bottom
        val minimumMargin = minOf(leftMargin, topMargin, rightMargin, bottomMargin)
        return when {
            minimumMargin <= 0.0 -> 1.0
            minimumMargin >= SAFE_MARGIN -> 0.0
            else -> (1.0 - minimumMargin / SAFE_MARGIN).coerceIn(0.0, 1.0)
        }
    }

    private fun NormalizedPoint.clampedToCompositionRange() = NormalizedPoint(
        x.coerceIn(0.08, 0.92),
        y.coerceIn(0.08, 0.92)
    )

    private const val SAFE_MARGIN = 0.09
    private const val SPEED_REFERENCE = 0.85
    private const val ACCEL_REFERENCE = 4.0
    private const val EXPOSURE_GAIN = 0.92
    private const val MIN_EXPOSURE_MULTIPLIER = 0.40
    private const val RETARGET_CONFIDENCE = 0.52
    private const val INITIAL_ACQUIRE_CONFIDENCE = 0.64
    private const val RETARGET_DISPLACEMENT = 0.012
    private const val RETARGET_SPEED = 0.10
    private const val LEAD_ROOM_GAIN = 0.055
    private const val VERTICAL_LEAD_GAIN = 0.025
    private const val MISS_CONFIDENCE_DECAY = 0.62
}
