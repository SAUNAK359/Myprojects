package ai.auraforge.director.director

import ai.auraforge.director.body.CropClass
import ai.auraforge.director.core.Point2
import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class RelationSpec(
    val objectLabel: String,
    val sideRule: String,
    val distance: Float,
    val weight: Float
)

data class PoseSpec(
    val centerX: Float,
    val centerY: Float,
    val scale: Float,
    val mirrorAllowed: Boolean,
    val joints: Map<Int, Point2>
)

data class ShotTemplate(
    val id: String,
    val name: String,
    val style: String,
    val requires: Set<String>,
    val forbids: Set<String>,
    val preferredObjects: Set<String>,
    val minimumPeople: Int,
    val aestheticPrior: Float,
    val difficulty: Float,
    val relation: RelationSpec?,
    val pose: PoseSpec,
    val supportedCrops: Set<CropClass>
)

class ShotTemplateRepository private constructor(
    val templates: List<ShotTemplate>
) {
    companion object {
        fun load(context: Context): ShotTemplateRepository {
            val raw = context.assets.open("shot_templates.json").bufferedReader().use { it.readText() }
            val root = JSONObject(raw)
            val arr = root.getJSONArray("templates")
            val result = buildList {
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    val poseJson = o.getJSONObject("pose")
                    val jointsJson = poseJson.getJSONObject("joints")
                    val joints = mutableMapOf<Int, Point2>()
                    val keys = jointsJson.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        val p = jointsJson.getJSONArray(key)
                        joints[key.toInt()] = Point2(
                            p.getDouble(0).toFloat(),
                            p.getDouble(1).toFloat(),
                            1f
                        )
                    }

                    val relation = if (o.has("relation")) {
                        val r = o.getJSONObject("relation")
                        RelationSpec(
                            objectLabel = r.getString("object"),
                            sideRule = r.getString("side"),
                            distance = r.getDouble("distance").toFloat(),
                            weight = r.getDouble("weight").toFloat()
                        )
                    } else null

                    val crops = if (o.has("supportedCrops")) {
                        o.getJSONArray("supportedCrops").toCropSet()
                    } else {
                        inferSupportedCrops(joints.keys)
                    }

                    add(
                        ShotTemplate(
                            id = o.getString("id"),
                            name = o.getString("name"),
                            style = o.getString("style"),
                            requires = o.getJSONArray("requires").toStringSet(),
                            forbids = o.getJSONArray("forbids").toStringSet(),
                            preferredObjects = o.getJSONArray("preferredObjects").toStringSet(),
                            minimumPeople = o.optInt("minimumPeople", 1),
                            aestheticPrior = o.getDouble("aestheticPrior").toFloat(),
                            difficulty = o.getDouble("difficulty").toFloat(),
                            relation = relation,
                            pose = PoseSpec(
                                centerX = poseJson.getDouble("centerX").toFloat(),
                                centerY = poseJson.getDouble("centerY").toFloat(),
                                scale = poseJson.getDouble("scale").toFloat(),
                                mirrorAllowed = poseJson.optBoolean("mirrorAllowed", true),
                                joints = joints
                            ),
                            supportedCrops = crops
                        )
                    )
                }
            }
            return ShotTemplateRepository(result)
        }

        private fun inferSupportedCrops(joints: Set<Int>): Set<CropClass> = buildSet {
            if (0 in joints) add(CropClass.FACE)
            if (11 in joints && 12 in joints) add(CropClass.HEAD_SHOULDERS)
            if ((13 in joints || 14 in joints) && 11 in joints && 12 in joints) add(CropClass.CHEST_UP)
            if (23 in joints && 24 in joints) add(CropClass.WAIST_UP)
            if (25 in joints || 26 in joints) add(CropClass.THREE_QUARTER)
            if (27 in joints || 28 in joints) {
                add(CropClass.FULL_BODY)
                add(CropClass.ENVIRONMENTAL)
            }
        }
    }
}

private fun JSONArray.toStringSet(): Set<String> = buildSet {
    for (i in 0 until length()) add(getString(i).lowercase())
}

private fun JSONArray.toCropSet(): Set<CropClass> = buildSet {
    for (i in 0 until length()) {
        runCatching { CropClass.valueOf(getString(i).uppercase()) }.getOrNull()?.let(::add)
    }
}
