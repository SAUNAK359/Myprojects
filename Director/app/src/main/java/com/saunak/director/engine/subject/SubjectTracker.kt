package com.saunak.director.engine.subject

import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

/**
 * Lightweight deterministic tracker for the primary photographic subject. It associates observations
 * using geometry, identity continuity and time; model-specific enrichment never changes track identity.
 */
class SubjectTracker(
    private val maxGapNs: Long = 900_000_000L,
    private val maxMisses: Int = 5,
    private val predictionHorizonMs: Long = 120L
) {
    init {
        require(maxGapNs > 0)
        require(maxMisses >= 0)
        require(predictionHorizonMs in 0..500)
    }

    private val ids = AtomicLong(0L)
    private var primary: InternalTrack? = null

    @Synchronized
    fun update(observation: SubjectObservation): SubjectTrack {
        val current = primary
        val next = if (current == null || !canAssociate(current, observation)) {
            InternalTrack.from(observation, ids.incrementAndGet(), predictionHorizonMs)
        } else {
            current.update(observation, predictionHorizonMs)
        }
        primary = next
        return next.public()
    }

    @Synchronized
    fun miss(nowNs: Long): SubjectTrack? {
        require(nowNs > 0)
        val current = primary ?: return null
        val ageNs = (nowNs - current.lastSeenNs).coerceAtLeast(0L)
        if (ageNs > maxGapNs || current.missedUpdates + 1 > maxMisses) {
            primary = null
            return null
        }
        val aged = current.copy(
            missedUpdates = current.missedUpdates + 1,
            confidence = (current.confidence * 0.78).coerceIn(0.0, 1.0),
            uncertainty = (current.uncertainty + 0.12).coerceIn(0.0, 1.0)
        )
        primary = aged
        return aged.public()
    }

    @Synchronized fun latest(): SubjectTrack? = primary?.public()
    @Synchronized fun clear() { primary = null }

    private fun canAssociate(track: InternalTrack, obs: SubjectObservation): Boolean {
        if (track.cameraId != obs.cameraId) return false
        val dtNs = obs.observedAtNs - track.lastSeenNs
        if (dtNs <= 0 || dtNs > maxGapNs) return false
        val iou = track.bounds.iou(obs.bounds)
        val distance = track.center.distanceTo(obs.bounds.center)
        val predictedDistance = track.kinematics.predictedCenter.distanceTo(obs.bounds.center)
        return iou >= MIN_IOU || minOf(distance, predictedDistance) <= MAX_CENTER_DISTANCE
    }

    private data class InternalTrack(
        val id: Long,
        val cameraId: String?,
        val bounds: NormalizedRect,
        val center: NormalizedPoint,
        val previousCenter: NormalizedPoint?,
        val previousVelocityX: Double,
        val previousVelocityY: Double,
        val kinematics: SubjectKinematics,
        val confidence: Double,
        val uncertainty: Double,
        val firstSeenNs: Long,
        val lastSeenNs: Long,
        val updateCount: Int,
        val missedUpdates: Int,
        val mask: SubjectMaskState?,
        val pose: SubjectPoseState?,
        val depth: SubjectDepthState?,
        val source: String
    ) {
        fun update(obs: SubjectObservation, horizonMs: Long): InternalTrack {
            val dt = ((obs.observedAtNs - lastSeenNs).coerceAtLeast(1L)) / 1_000_000_000.0
            val rawVx = (obs.bounds.center.x - center.x) / dt
            val rawVy = (obs.bounds.center.y - center.y) / dt
            val alpha = if (updateCount <= 1) 1.0 else VELOCITY_ALPHA
            val vx = previousVelocityX * (1.0 - alpha) + rawVx * alpha
            val vy = previousVelocityY * (1.0 - alpha) + rawVy * alpha
            val ax = (vx - previousVelocityX) / dt
            val ay = (vy - previousVelocityY) / dt
            val horizonSec = horizonMs / 1000.0
            val predicted = NormalizedPoint(
                obs.bounds.center.x + vx * horizonSec + 0.5 * ax * horizonSec * horizonSec,
                obs.bounds.center.y + vy * horizonSec + 0.5 * ay * horizonSec * horizonSec
            ).clamped()
            val enrichedConfidence = listOfNotNull(
                obs.confidence,
                obs.faceConfidence,
                obs.mask?.confidence,
                obs.pose?.confidence,
                obs.depth?.confidence
            ).average().coerceIn(0.0, 1.0)
            val uncertainty = (1.0 - enrichedConfidence + motionUncertainty(vx, vy, ax, ay)).coerceIn(0.0, 1.0)
            return copy(
                bounds = obs.bounds,
                center = obs.bounds.center,
                previousCenter = center,
                previousVelocityX = vx,
                previousVelocityY = vy,
                kinematics = SubjectKinematics(vx, vy, ax, ay, predicted, horizonMs),
                confidence = enrichedConfidence,
                uncertainty = uncertainty,
                lastSeenNs = obs.observedAtNs,
                updateCount = updateCount + 1,
                missedUpdates = 0,
                mask = obs.mask ?: mask,
                pose = obs.pose ?: pose,
                depth = obs.depth ?: depth,
                source = obs.source
            )
        }

        fun public() = SubjectTrack(
            id = id,
            cameraId = cameraId,
            bounds = bounds,
            center = center,
            kinematics = kinematics,
            confidence = confidence,
            uncertainty = uncertainty,
            firstSeenNs = firstSeenNs,
            lastSeenNs = lastSeenNs,
            updateCount = updateCount,
            missedUpdates = missedUpdates,
            mask = mask,
            pose = pose,
            depth = depth,
            source = source
        )

        companion object {
            fun from(obs: SubjectObservation, id: Long, horizonMs: Long): InternalTrack {
                val confidence = listOfNotNull(
                    obs.confidence,
                    obs.faceConfidence,
                    obs.mask?.confidence,
                    obs.pose?.confidence,
                    obs.depth?.confidence
                ).average().coerceIn(0.0, 1.0)
                val center = obs.bounds.center
                return InternalTrack(
                    id = id,
                    cameraId = obs.cameraId,
                    bounds = obs.bounds,
                    center = center,
                    previousCenter = null,
                    previousVelocityX = 0.0,
                    previousVelocityY = 0.0,
                    kinematics = SubjectKinematics(0.0, 0.0, 0.0, 0.0, center, horizonMs),
                    confidence = confidence,
                    uncertainty = (1.0 - confidence).coerceIn(0.0, 1.0),
                    firstSeenNs = obs.observedAtNs,
                    lastSeenNs = obs.observedAtNs,
                    updateCount = 1,
                    missedUpdates = 0,
                    mask = obs.mask,
                    pose = obs.pose,
                    depth = obs.depth,
                    source = obs.source
                )
            }

            private fun motionUncertainty(vx: Double, vy: Double, ax: Double, ay: Double): Double {
                val speed = kotlin.math.hypot(vx, vy)
                val acceleration = kotlin.math.hypot(ax, ay)
                return (0.05 * speed + 0.012 * acceleration).coerceIn(0.0, 0.35)
            }
        }
    }

    companion object {
        private const val MIN_IOU = 0.08
        private const val MAX_CENTER_DISTANCE = 0.28
        private const val VELOCITY_ALPHA = 0.62
    }
}
