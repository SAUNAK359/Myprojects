package ai.auraforge.director.ui

import ai.auraforge.director.core.DirectorFrameState
import ai.auraforge.director.core.DirectorLockState
import ai.auraforge.director.core.Point2
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class DirectorOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var state = DirectorFrameState()
    private var mirrorX = false

    private val ghostPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(113, 247, 255)
        strokeWidth = dp(2.2f)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        pathEffect = DashPathEffect(floatArrayOf(dp(9f), dp(7f)), 0f)
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(145, 255, 178)
        style = Paint.Style.FILL
    }
    private val livePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(110, 255, 255, 255)
        strokeWidth = dp(1.1f)
        style = Paint.Style.STROKE
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(85, 255, 255, 255)
        strokeWidth = dp(0.8f)
    }
    private val anchorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(175, 113, 247, 255)
        strokeWidth = dp(1.5f)
        style = Paint.Style.STROKE
    }
    private val lockPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    fun setState(newState: DirectorFrameState, mirrored: Boolean) {
        state = newState
        mirrorX = mirrored
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scene = state.scene ?: return
        if (state.gridEnabled) drawGrid(canvas)

        val locked = state.alignment.state == DirectorLockState.LOCKED
        ghostPaint.color = if (locked) Color.rgb(145, 255, 178) else Color.rgb(113, 247, 255)
        jointPaint.color = ghostPaint.color

        state.plan?.targetPoses?.forEach { target ->
            drawSkeleton(canvas, target.joints, ghostPaint, jointPaint)
        }

        scene.poses.forEach { pose ->
            drawSkeleton(canvas, pose.landmarks.mapIndexed { idx, p -> idx to p }.toMap(), livePaint, null)
        }

        state.plan?.relation?.let { rel ->
            scene.primary(rel.objectLabel)?.let { drawEntityAnchor(canvas, it.bbox, rel.objectLabel.uppercase()) }
        }

        drawLockMeter(canvas)
    }

    private fun drawGrid(canvas: Canvas) {
        val x1 = width / 3f
        val x2 = width * 2f / 3f
        val y1 = height / 3f
        val y2 = height * 2f / 3f
        canvas.drawLine(x1, 0f, x1, height.toFloat(), gridPaint)
        canvas.drawLine(x2, 0f, x2, height.toFloat(), gridPaint)
        canvas.drawLine(0f, y1, width.toFloat(), y1, gridPaint)
        canvas.drawLine(0f, y2, width.toFloat(), y2, gridPaint)
    }

    private fun drawSkeleton(
        canvas: Canvas,
        joints: Map<Int, Point2>,
        linePaint: Paint,
        pointPaint: Paint?
    ) {
        CONNECTIONS.forEach { (a, b) ->
            val pa = joints[a] ?: return@forEach
            val pb = joints[b] ?: return@forEach
            if (pa.confidence < 0.28f || pb.confidence < 0.28f) return@forEach
            val va = map(pa)
            val vb = map(pb)
            canvas.drawLine(va.first, va.second, vb.first, vb.second, linePaint)
        }
        pointPaint?.let { paint ->
            IMPORTANT_JOINTS.forEach { idx ->
                val p = joints[idx] ?: return@forEach
                val v = map(p)
                canvas.drawCircle(v.first, v.second, dp(if (idx == 0) 4.8f else 3.4f), paint)
            }
        }
    }

    private fun drawEntityAnchor(canvas: Canvas, normalized: RectF, label: String) {
        val p1 = map(Point2(normalized.left, normalized.top))
        val p2 = map(Point2(normalized.right, normalized.bottom))
        val rect = RectF(minOf(p1.first, p2.first), minOf(p1.second, p2.second), maxOf(p1.first, p2.first), maxOf(p1.second, p2.second))
        anchorPaint.pathEffect = DashPathEffect(floatArrayOf(dp(6f), dp(6f)), 0f)
        canvas.drawRoundRect(rect, dp(14f), dp(14f), anchorPaint)
        lockPaint.color = Color.argb(210, 113, 247, 255)
        lockPaint.textSize = dp(9f)
        canvas.drawText(label, rect.left + rect.width() / 2f, rect.top - dp(7f), lockPaint)
    }

    private fun drawLockMeter(canvas: Canvas) {
        val a = state.alignment
        if (state.plan == null) return
        val x = width / 2f
        val y = height * 0.18f
        val r = dp(29f)
        val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3f)
            strokeCap = Paint.Cap.ROUND
            color = Color.argb(90, 255, 255, 255)
        }
        canvas.drawCircle(x, y, r, ring)
        ring.color = when (a.state) {
            DirectorLockState.LOCKED -> Color.rgb(145, 255, 178)
            DirectorLockState.NEAR_LOCK, DirectorLockState.LOCK_CANDIDATE -> Color.rgb(255, 210, 122)
            else -> Color.rgb(113, 247, 255)
        }
        val sweep = 360f * a.totalScore.coerceIn(0f, 1f)
        canvas.drawArc(RectF(x - r, y - r, x + r, y + r), -90f, sweep, false, ring)
        lockPaint.color = Color.WHITE
        lockPaint.textSize = dp(9f)
        canvas.drawText(if (a.state == DirectorLockState.LOCKED) "LOCKED" else "${(a.totalScore * 100).toInt()}%", x, y + dp(3f), lockPaint)
    }

    private fun map(p: Point2): Pair<Float, Float> {
        val scene = state.scene ?: return 0f to 0f
        val iw = scene.imageWidth.toFloat().coerceAtLeast(1f)
        val ih = scene.imageHeight.toFloat().coerceAtLeast(1f)
        val scale = max(width / iw, height / ih)
        val scaledW = iw * scale
        val scaledH = ih * scale
        val ox = (width - scaledW) / 2f
        val oy = (height - scaledH) / 2f
        val nx = if (mirrorX) 1f - p.x else p.x
        return ox + nx * iw * scale to oy + p.y * ih * scale
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density

    companion object {
        private val IMPORTANT_JOINTS = listOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        private val CONNECTIONS = listOf(
            11 to 12,
            11 to 13, 13 to 15,
            12 to 14, 14 to 16,
            11 to 23, 12 to 24,
            23 to 24,
            23 to 25, 25 to 27,
            24 to 26, 26 to 28,
            0 to 11, 0 to 12
        )
    }
}
