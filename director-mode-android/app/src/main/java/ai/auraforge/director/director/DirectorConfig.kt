package ai.auraforge.director.director

import android.content.Context
import org.json.JSONObject

data class LockThresholds(
    val aligning: Float,
    val nearLock: Float,
    val candidate: Float,
    val unlock: Float,
    val technicalMin: Float,
    val candidateHoldMs: Long,
    val captureCooldownMs: Long
)

data class AlignmentWeights(
    val pose: Float,
    val framing: Float,
    val relation: Float,
    val head: Float,
    val composition: Float
)

data class CaptureThresholds(
    val sharpnessMin: Float,
    val eyesOpenMin: Float,
    val motionMax: Float,
    val exposureMin: Float
)

data class DirectorTuning(
    val minimumExpectedGain: Float,
    val planCommitScore: Float,
    val replanIntervalMs: Long
)

data class DirectorConfig(
    val lock: LockThresholds,
    val weights: AlignmentWeights,
    val capture: CaptureThresholds,
    val director: DirectorTuning
) {
    companion object {
        fun load(context: Context): DirectorConfig {
            val raw = context.assets.open("director_config.json").bufferedReader().use { it.readText() }
            val root = JSONObject(raw)
            val lock = root.getJSONObject("lock")
            val w = root.getJSONObject("weights")
            val c = root.getJSONObject("capture")
            val d = root.getJSONObject("director")
            return DirectorConfig(
                lock = LockThresholds(
                    lock.getDouble("aligning").toFloat(),
                    lock.getDouble("nearLock").toFloat(),
                    lock.getDouble("candidate").toFloat(),
                    lock.getDouble("unlock").toFloat(),
                    lock.getDouble("technicalMin").toFloat(),
                    lock.getLong("candidateHoldMs"),
                    lock.getLong("captureCooldownMs")
                ),
                weights = AlignmentWeights(
                    w.getDouble("pose").toFloat(),
                    w.getDouble("framing").toFloat(),
                    w.getDouble("relation").toFloat(),
                    w.getDouble("head").toFloat(),
                    w.getDouble("composition").toFloat()
                ),
                capture = CaptureThresholds(
                    c.getDouble("sharpnessMin").toFloat(),
                    c.getDouble("eyesOpenMin").toFloat(),
                    c.getDouble("motionMax").toFloat(),
                    c.getDouble("exposureMin").toFloat()
                ),
                director = DirectorTuning(
                    d.getDouble("minimumExpectedGain").toFloat(),
                    d.getDouble("planCommitScore").toFloat(),
                    d.getLong("replanIntervalMs")
                )
            )
        }
    }
}
