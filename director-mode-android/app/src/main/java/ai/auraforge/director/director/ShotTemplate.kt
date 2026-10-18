package ai.auraforge.director.director

import ai.auraforge.director.core.Point2
import android.content.Context
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
    val pose: PoseSpec
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
                            )
                        )
                    )
                }
            }
            return ShotTemplateRepository(result)
        }
    }
}

private fun org.json.JSONArray.toStringSet(): Set<String> = buildSet {
    for (i in 0 until length()) add(getString(i).lowercase())
}
