package com.saunak.director.director

import android.graphics.Bitmap
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.SceneBeliefGraph
import com.saunak.director.engine.experts.DepthAnythingV2Expert
import com.saunak.director.engine.experts.DepthSceneInterpreter
import com.saunak.director.engine.experts.FaceDetection
import com.saunak.director.engine.experts.PersonSegmentationExpert
import com.saunak.director.engine.experts.RTMPoseExpert
import com.saunak.director.engine.experts.SubjectDepthEvidence
import com.saunak.director.engine.experts.YuNetFaceExpert
import com.saunak.director.engine.subject.NormalizedRect
import com.saunak.director.engine.subject.SubjectDepthState
import com.saunak.director.engine.subject.SubjectMaskState
import com.saunak.director.engine.subject.SubjectObservation
import com.saunak.director.engine.subject.SubjectPoseState
import com.saunak.director.engine.subject.SubjectTrack
import com.saunak.director.processing.TechnicalQuality
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Shutter-time perception coordinator. YuNet establishes inexpensive human identity evidence first.
 * Optional segmentation, pose and depth experts then enrich the SAME timestamped observation in
 * parallel. Model failure or absence never invalidates the face-based subject track.
 */
class SierraPerceptionFabric(
    private val engine: SierraEngine = SierraEngine.get(),
    private val faceExpert: YuNetFaceExpert = YuNetFaceExpert(engine),
    private val segmentationExpert: PersonSegmentationExpert = PersonSegmentationExpert(engine),
    private val poseExpert: RTMPoseExpert = RTMPoseExpert(engine),
    private val depthExpert: DepthAnythingV2Expert = DepthAnythingV2Expert(engine)
) {
    data class Snapshot(
        val signals: SierraSceneSignals,
        val faces: List<FaceDetection>?,
        val subject: SubjectTrack?,
        val segmentationUsed: Boolean,
        val poseUsed: Boolean,
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
        val effectiveCameraId = cameraId ?: "lens:$lensFacing"
        val faces = withTimeoutOrNull(FACE_BUDGET_MS) { faceExpert.detect(frame).getOrNull() }
        val preliminary = SierraSceneAnalyzer.analyze(frame, quality, lensFacing, faces)
        val primaryFace = faces?.maxByOrNull { it.area * it.score }

        if (primaryFace == null) {
            val aged = engine.subjects.miss(now)
            publishBeliefs(preliminary, faces, aged, quality, now)
            engine.beliefs.prune(now)
            return Snapshot(preliminary, faces, aged, false, false, false, now)
        }

        val proxyBounds = subjectBoundsFromFace(primaryFace, frame.width, frame.height)
        val subjectVoI = subjectValueOfInformation(preliminary, primaryFace, thermalPressure)
        val depthVoI = depthValueOfInformation(preliminary, zoomRatio, thermalPressure)

        data class Enrichment(
            val mask: SubjectMaskState?,
            val pose: SubjectPoseState?,
            val depth: SubjectDepthEvidence?
        )

        val enrichment = coroutineScope {
            val maskDeferred = async {
                if (subjectVoI >= SEGMENTATION_VOI_THRESHOLD && segmentationExpert.isProvisioned()) {
                    withTimeoutOrNull(SEGMENTATION_BUDGET_MS) {
                        segmentationExpert.segment(frame, proxyBounds).getOrNull()
                    }
                } else null
            }
            val poseDeferred = async {
                if (subjectVoI >= POSE_VOI_THRESHOLD && poseExpert.isProvisioned()) {
                    withTimeoutOrNull(POSE_BUDGET_MS) {
                        poseExpert.estimate(frame, proxyBounds).getOrNull()
                    }
                } else null
            }
            val depthDeferred = async {
                if (depthVoI >= DEPTH_VOI_THRESHOLD && depthExpert.isProvisioned()) {
                    withTimeoutOrNull(DEPTH_BUDGET_MS) {
                        depthExpert.estimate(frame).getOrNull()?.let { depth ->
                            DepthSceneInterpreter.subjectSeparation(depth, faces.orEmpty())
                        }
                    }
                } else null
            }
            Enrichment(maskDeferred.await(), poseDeferred.await(), depthDeferred.await())
        }

        val signals = if (enrichment.depth != null) {
            SierraSceneAnalyzer.analyze(frame, quality, lensFacing, faces, enrichment.depth)
        } else preliminary

        val fusedBounds = fuseSubjectBounds(proxyBounds, enrichment.mask, enrichment.pose)
        val subject = engine.subjects.update(
            SubjectObservation(
                observedAtNs = now,
                cameraId = effectiveCameraId,
                bounds = fusedBounds,
                confidence = primaryFace.score.toDouble(),
                faceConfidence = primaryFace.score.toDouble(),
                mask = enrichment.mask,
                pose = enrichment.pose,
                depth = enrichment.depth?.toSubjectDepth(),
                source = buildString {
                    append("yunet")
                    if (enrichment.mask != null) append("+mask")
                    if (enrichment.pose != null) append("+pose")
                    if (enrichment.depth != null) append("+depth")
                }
            )
        )

        publishBeliefs(signals, faces, subject, quality, now)
        enrichment.depth?.let {
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

        return Snapshot(
            signals = signals,
            faces = faces,
            subject = subject,
            segmentationUsed = enrichment.mask != null,
            poseUsed = enrichment.pose != null,
            depthUsed = enrichment.depth != null,
            capturedAtNs = now
        )
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
            track.mask?.let { mask ->
                engine.beliefs.put(
                    SceneBeliefGraph.SUBJECT_MASK,
                    Belief(mask, mask.confidence, 1.0 - mask.confidence, now, mask.source, MASK_HALF_LIFE_NS)
                )
            }
            track.pose?.let { pose ->
                engine.beliefs.put(
                    SceneBeliefGraph.SUBJECT_POSE,
                    Belief(pose, pose.confidence, 1.0 - pose.confidence, now, pose.source, POSE_HALF_LIFE_NS)
                )
            }
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

    private fun fuseSubjectBounds(
        proxy: NormalizedRect,
        mask: SubjectMaskState?,
        pose: SubjectPoseState?
    ): NormalizedRect {
        val candidates = buildList {
            add(proxy to 0.34)
            if (mask != null && mask.confidence >= 0.28) add(mask.bounds to (0.42 * mask.confidence))
            if (pose != null && pose.confidence >= 0.22) add(pose.bounds to (0.34 * pose.confidence))
        }
        val total = candidates.sumOf { it.second }.coerceAtLeast(1e-6)
        return NormalizedRect(
            left = candidates.sumOf { it.first.left * it.second } / total,
            top = candidates.sumOf { it.first.top * it.second } / total,
            right = candidates.sumOf { it.first.right * it.second } / total,
            bottom = candidates.sumOf { it.first.bottom * it.second } / total
        ).clamped()
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

    private fun subjectValueOfInformation(
        signals: SierraSceneSignals,
        face: FaceDetection,
        thermalPressure: Double
    ): Double {
        if (!signals.primaryHumanSubject) return 0.0
        val branchValue = when (signals.branch) {
            "portrait", "selfie" -> 1.0
            "group" -> 0.74
            else -> 0.52
        }
        val sizeValue = (signals.faceCoverage / 0.12).coerceIn(0.0, 1.0)
        val confidenceValue = face.score.toDouble().coerceIn(0.0, 1.0)
        return (0.48 * branchValue + 0.28 * sizeValue + 0.24 * confidenceValue -
            0.50 * thermalPressure.coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)
    }

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
        private const val SEGMENTATION_BUDGET_MS = 130L
        private const val POSE_BUDGET_MS = 175L
        private const val DEPTH_BUDGET_MS = 300L
        private const val SEGMENTATION_VOI_THRESHOLD = 0.42
        private const val POSE_VOI_THRESHOLD = 0.55
        private const val DEPTH_VOI_THRESHOLD = 0.56
        private const val SCENE_HALF_LIFE_NS = 650_000_000L
        private const val HUMAN_HALF_LIFE_NS = 350_000_000L
        private const val SUBJECT_HALF_LIFE_NS = 520_000_000L
        private const val TRAJECTORY_HALF_LIFE_NS = 240_000_000L
        private const val MASK_HALF_LIFE_NS = 320_000_000L
        private const val POSE_HALF_LIFE_NS = 260_000_000L
        private const val DEPTH_HALF_LIFE_NS = 500_000_000L
        private const val QUALITY_HALF_LIFE_NS = 250_000_000L
    }
}
