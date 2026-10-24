package com.saunak.director.engine.geometry

import java.util.ArrayDeque
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs

enum class CoordinateSpace {
    SENSOR_RAW,
    PRECORRECTION_ACTIVE_ARRAY,
    ACTIVE_ARRAY,
    CROP_REGION,
    ANALYSIS,
    MODEL_INPUT,
    PREVIEW,
    DISPLAY
}

data class Point2D(val x: Double, val y: Double)

/** Row-major projective 3x3 transform. */
data class Transform3x3(val m: DoubleArray) {
    init { require(m.size == 9) }

    fun map(point: Point2D): Point2D {
        val w = m[6] * point.x + m[7] * point.y + m[8]
        require(abs(w) > EPS) { "Non-invertible homogeneous point" }
        return Point2D(
            (m[0] * point.x + m[1] * point.y + m[2]) / w,
            (m[3] * point.x + m[4] * point.y + m[5]) / w
        )
    }

    /** this followed by next. */
    fun then(next: Transform3x3): Transform3x3 = multiply(next, this)

    fun inverse(): Transform3x3 {
        val a = m[0]; val b = m[1]; val c = m[2]
        val d = m[3]; val e = m[4]; val f = m[5]
        val g = m[6]; val h = m[7]; val i = m[8]
        val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        require(abs(det) > EPS) { "Transform is singular" }
        val inv = doubleArrayOf(
            e * i - f * h, c * h - b * i, b * f - c * e,
            f * g - d * i, a * i - c * g, c * d - a * f,
            d * h - e * g, b * g - a * h, a * e - b * d
        )
        for (index in inv.indices) inv[index] /= det
        return Transform3x3(inv)
    }

    companion object {
        private const val EPS = 1e-10
        val IDENTITY = Transform3x3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        fun scale(sx: Double, sy: Double) = Transform3x3(doubleArrayOf(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0))
        fun translate(tx: Double, ty: Double) = Transform3x3(doubleArrayOf(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0))

        fun multiply(a: Transform3x3, b: Transform3x3): Transform3x3 {
            val out = DoubleArray(9)
            for (r in 0..2) for (c in 0..2) {
                var sum = 0.0
                for (k in 0..2) sum += a.m[r * 3 + k] * b.m[k * 3 + c]
                out[r * 3 + c] = sum
            }
            return Transform3x3(out)
        }
    }
}

/**
 * Explicit coordinate-space graph. Any detector/mask/depth result must declare the space it uses.
 * Edges are stored bidirectionally to prevent ad-hoc UI/sensor conversions throughout the app.
 */
class CoordinateGraph {
    private val edges = ConcurrentHashMap<Pair<CoordinateSpace, CoordinateSpace>, Transform3x3>()

    fun clear() = edges.clear()

    fun connect(from: CoordinateSpace, to: CoordinateSpace, transform: Transform3x3) {
        require(from != to)
        edges[from to to] = transform
        edges[to to from] = transform.inverse()
    }

    fun transform(from: CoordinateSpace, to: CoordinateSpace): Transform3x3? {
        if (from == to) return Transform3x3.IDENTITY
        edges[from to to]?.let { return it }

        data class Node(val space: CoordinateSpace, val transform: Transform3x3)
        val queue = ArrayDeque<Node>()
        val visited = mutableSetOf(from)
        queue.add(Node(from, Transform3x3.IDENTITY))

        while (queue.isNotEmpty()) {
            val current = queue.removeFirst()
            for ((edge, edgeTransform) in edges) {
                if (edge.first != current.space || edge.second in visited) continue
                val chained = current.transform.then(edgeTransform)
                if (edge.second == to) return chained
                visited += edge.second
                queue.add(Node(edge.second, chained))
            }
        }
        return null
    }

    fun map(point: Point2D, from: CoordinateSpace, to: CoordinateSpace): Point2D =
        requireNotNull(transform(from, to)) { "No coordinate path from $from to $to" }.map(point)
}
