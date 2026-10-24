package com.saunak.director.director

import android.graphics.Bitmap
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.SceneBeliefGraph
import com.saunak.director.engine.experts.DepthAnythingV2Expert
import com.saunak.director.engine.experts.DepthSceneInterpreter
import com.saunak.director.engine.experts.FaceDetection
import com.saunak.director.engine.experts.SubjectDepthEvidence
import com.saunak.director.engine.experts.YuNetFaceExpert
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectDepthState
import com.saunak.director.engine.subject.SubjectObservation
import com.saunak.director.engine.subject.SubjectTrack
import com.saunak.director.processing.TechnicalQuality
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Shutter-time perception coordinator. Cheap human semantics run first; expensive depth is invoked
 * only when its expected photographic value is high enough and a verified model already exists.
 * Every human observation feeds the application-scoped SubjectTracker before capture planning.
 */
class SierraPerceptionFabric(
    private val engine: SierraEngine = SierraEngine.get(),
    private val faceExpert: YuNetFaceExpert = YuNetFaceExpert(engine),
    private val depthExpert: DepthAnythingV2Expert = DepthAnythingV2Expert(engine)
) {
    data class Snapshot(
        val signals: SierraSceneSignals,
        val faces: List<FaceDetection>?,
        val subject: SubjectTrack?,
        val depthUsed: Boolean,
        val capturedAtNs: Long
    )

    suspend fun analyze(
        frame: Bitmap,
        quality: TechnicalQuality,
        lensFacing: Int,
        zoomRatio: Float,
        thermalPressure: Double,
        cameraId: String? = null
    ): Snapshot {
        val now = System.nanoTime()
        val faces = withTimeoutOrNull(FACE_BUDGET_MS) { faceExpert.detect(frame).getOrNull() }
        val preliminary = SierraSceneAnalyzer.analyze(frame, quality, lensFacing, faces)

        val depthValue = depthValueOfInformation(preliminary, zoomRatio, thermalPressure)
        val depthEvidence = if (depthValue >= DEPTH_VOI_THRESHOLD && depthExpert.isProvisioned()) {
            withTimeoutOrNull(DEPTH_BUDGET_MS) {
                depthExpert.estimate(frame).getOrNull()?.let { depth ->
                    DepthSceneInterpreter.subjectSeparation(depth, faces.orEmpty())
                }
            }
        } else null

        val signals = if (depthEvidence != null) {
            SierraSceneAnalyzer.analyze(frame, quality, lensFacing, faces, depthEvidence)
        } else preliminary

        val primaryFace = faces?.maxByOrNull { it.area * it.score }
        val subject = if (primaryFace != null) {
            engine.subjects.update(
                SubjectObservation(
                    observedAtNs = now,
                    cameraId = cameraId,
                    bounds = subjectBoundsFromFace(primaryFace, frame.width, frame.height),
                    confidence = primaryFace.score.toDouble(),
                    faceConfidence = primaryFace.score.toDouble(),
                    depth = depthEvidence?.toSubjectDepth(),
                    source = "yunet_temporal_subject"
                )
            )
        } else {
            engine.subjects.miss(now)
        }

        publishBeliefs(signals, faces, subject, quality, now)
        depthEvidence?.let {
            engine.beliefs.put(
                SceneBeliefGraph.DEPTH,
                Belief(
                    value = it,
                    confidence = it.confidence,
                    uncertainty = 1.0 - it.confidence,
                    timestampNs = now,
                    source = it.source,
                    halfLifeNs = DEPTH_HALF_LIFE_NS
                )
            )
        }
        engine.beliefs.prune(now)

        return Snapshot(signals, faces, subject, depthEvidence != null, now)
    }

    private fun publishBeliefs(
        signals: SierraSceneSignals,
        faces: List<FaceDetection>?,
        subject: SubjectTrack?,
        quality: TechnicalQuality,
        now: Long
    ) {
        engine.beliefs.put(
            SceneBeliefGraph.SCENE,
            Belief(signals.branch, signals.confidence, 1.0 - signals.confidence, now, "sierra_perception", SCENE_HALF_LIFE_NS)
        )
        engine.beliefs.put(
            SceneBeliefGraph.PERSON_COUNT,
            Belief(signals.faceCount, if (signals.faceSource == "yunet") 0.96 else 0.62, 0.0, now, signals.faceSource, HUMAN_HALF_LIFE_NS)
        )
        faces?.maxByOrNull { it.area * it.score }?.let { face ->
            engine.beliefs.put(
                SceneBeliefGraph.PRIMARY_FACE,
                Belief(face, face.score.toDouble(), 1.0 - face.score, now, "yunet", HUMAN_HALF_LIFE_NS)
            )
        }
        subject?.let { track ->
            engine.beliefs.put(
                SceneBeliefGraph.PRIMARY_SUBJECT,
                Belief(track, track.confidence, track.uncertainty, now, track.source, SUBJECT_HALF_LIFE_NS)
            )
            engine.beliefs.put(
                SceneBeliefGraph.SUBJECT_TRAJECTORY,
                Belief(track.kinematics, track.confidence, track.uncertainty, now, "subject_tracker", TRAJECTORY_HALF_LIFE_NS)
            )
            track.depth?.let { depth ->
                engine.beliefs.put(
                    SceneBeliefGraph.SUBJECT_DEPTH,
                    Belief(depth, depth.confidence, 1.0 - depth.confidence, now, depth.source, DEPTH_HALF_LIFE_NS)
                )
            }
        }
        engine.beliefs.put(
            SceneBeliefGraph.HIGHLIGHT_RISK,
            Belief(quality.brightClipFraction, 0.9, 0.1, now, "technical_quality", QUALITY_HALF_LIFE_NS)
        )
    }

    private fun subjectBoundsFromFace(face: FaceDetection, frameWidth: Int, frameHeight: Int): NormalizedRect {
        val cx = face.x + face.width * 0.5f
        val left = cx - face.width * 1.20f
        val right = cx + face.width * 1.20f
        val top = face.y - face.height * 0.30f
        val bottom = face.y + face.height * 4.10f
        return NormalizedRect.fromPixels(
            x = left.toDouble(),
            y = top.toDouble(),
            width = (right - left).toDouble().coerceAtLeast(1.0),
            height = (bottom - top).toDouble().coerceAtLeast(1.0),
            frameWidth = frameWidth,
            frameHeight = frameHeight
        )
    }

    private fun SubjectDepthEvidence.toSubjectDepth() = SubjectDepthState(
        relativeInverseDepth = subjectInverseDepth,
        foregroundSeparation = separation,
        confidence = confidence,
        source = source
    )

    private fun depthValueOfInformation(
        signals: SierraSceneSignals,
        zoomRatio: Float,
        thermalPressure: Double
    ): Double {
        if (!signals.primaryHumanSubject || signals.faceCount <= 0) return 0.0
        val branchValue = when (signals.branch) {
            "portrait", "selfie" -> 0.88
            "group" -> 0.65
            else -> 0.35
        }
        val faceValue = (signals.faceCoverage / 0.16).coerceIn(0.0, 1.0)
        val zoomValue = ((zoomRatio - 1f) / 2f).toDouble().coerceIn(0.0, 1.0)
        val thermalPenalty = thermalPressure.coerceIn(0.0, 1.0) * 0.55
        return (0.62 * branchValue + 0.25 * faceValue + 0.13 * zoomValue - thermalPenalty).coerceIn(0.0, 1.0)
    }

    companion object {
        private const val FACE_BUDGET_MS = 160L
        private const val DEPTH_BUDGET_MS = 300L
        private const val DEPTH_VOI_THRESHOLD = 0.56
        private const val SCENE_HALF_LIFE_NS = 650_000_000L
        private const val HUMAN_HALF_LIFE_NS = 350_000_000L
        private const val SUBJECT_HALF_LIFE_NS = 520_000_000L
        private const val TRAJECTORY_HALF_LIFE_NS = 240_000_000L
        private const val DEPTH_HALF_LIFE_NS = 500_000_000L
        private const val QUALITY_HALF_LIFE_NS = 250_000_000L
    }
}
