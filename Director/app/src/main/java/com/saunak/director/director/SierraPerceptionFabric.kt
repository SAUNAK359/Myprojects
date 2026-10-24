package com.saunak.director.director

import android.graphics.Bitmap
import com.saunak.director.engine.SierraEngine
import com.saunak.director.engine.belief.Belief
import com.saunak.director.engine.belief.SceneBeliefGraph
import com.saunak.director.engine.experts.DepthAnythingV2Expert
import com.saunak.director.engine.experts.DepthSceneInterpreter
import com.saunak.director.engine.experts.FaceDetection
import com.saunak.director.engine.experts.YuNetFaceExpert
import com.saunak.director.processing.TechnicalQuality
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Shutter-time perception coordinator. Cheap human semantics run first; expensive depth is invoked
 * only when its expected photographic value is high enough and a verified model already exists.
 */
class SierraPerceptionFabric(
    private val engine: SierraEngine = SierraEngine.get(),
    private val faceExpert: YuNetFaceExpert = YuNetFaceExpert(engine),
    private val depthExpert: DepthAnythingV2Expert = DepthAnythingV2Expert(engine)
) {
    data class Snapshot(
        val signals: SierraSceneSignals,
        val faces: List<FaceDetection>?,
        val depthUsed: Boolean,
        val capturedAtNs: Long
    )

    suspend fun analyze(
        frame: Bitmap,
        quality: TechnicalQuality,
        lensFacing: Int,
        zoomRatio: Float,
        thermalPressure: Double
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

        publishBeliefs(signals, faces, quality, now)
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

        return Snapshot(signals, faces, depthEvidence != null, now)
    }

    private fun publishBeliefs(
        signals: SierraSceneSignals,
        faces: List<FaceDetection>?,
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
        engine.beliefs.put(
            SceneBeliefGraph.HIGHLIGHT_RISK,
            Belief(quality.brightClipFraction, 0.9, 0.1, now, "technical_quality", QUALITY_HALF_LIFE_NS)
        )
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
        private const val DEPTH_BUDGET_MS = 300L
        private const val DEPTH_VOI_THRESHOLD = 0.56
        private const val SCENE_HALF_LIFE_NS = 650_000_000L
        private const val HUMAN_HALF_LIFE_NS = 350_000_000L
        private const val DEPTH_HALF_LIFE_NS = 500_000_000L
        private const val QUALITY_HALF_LIFE_NS = 250_000_000L
    }
}
