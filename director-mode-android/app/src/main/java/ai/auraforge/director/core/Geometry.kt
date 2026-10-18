package ai.auraforge.director.core

import android.graphics.RectF
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min

object Geometry {
    fun bbox(points: List<Point2>, minConfidence: Float = 0.25f): RectF {
        val valid = points.filter { it.confidence >= minConfidence }
        if (valid.isEmpty()) return RectF(0f, 0f, 0f, 0f)
        return RectF(
            valid.minOf { it.x },
            valid.minOf { it.y },
            valid.maxOf { it.x },
            valid.maxOf { it.y }
        ).clamp01()
    }

    fun angleDeg(a: Point2, b: Point2, c: Point2): Float {
        val ab = atan2(a.y - b.y, a.x - b.x)
        val cb = atan2(c.y - b.y, c.x - b.x)
        var d = abs((ab - cb) * 180f / PI.toFloat())
        if (d > 180f) d = 360f - d
        return d
    }

    fun centerDistance(a: RectF, b: RectF): Float {
        val dx = a.centerX() - b.centerX()
        val dy = a.centerY() - b.centerY()
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val l = max(a.left, b.left)
        val t = max(a.top, b.top)
        val r = min(a.right, b.right)
        val bt = min(a.bottom, b.bottom)
        if (r <= l || bt <= t) return 0f
        val inter = (r - l) * (bt - t)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t.coerceIn(0f, 1f)
}
