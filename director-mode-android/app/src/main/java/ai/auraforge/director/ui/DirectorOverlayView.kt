package ai.auraforge.director.ui

import ai.auraforge.director.core.CompositionGuide
import ai.auraforge.director.core.DirectorFrameState
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.PlanReadiness
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Director's live photographer HUD. Pose skeletons are intentionally not rendered. Human landmarks
 * remain private execution evidence for focus/framing/moment decisions, while the user sees only
 * useful photographic composition and readiness information.
 */
class DirectorOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var state = DirectorFrameState()
    private var plan: PhotoPlan? = null
    private var readiness = PlanReadiness(false, 0f, "Analyzing")
    private var mirrorX = false

    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(105, 255, 255, 255)
        strokeWidth = dp(0.8f)
        style = Paint.Style.STROKE
    }
    private val horizonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = dp(1.5f)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val focusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 255, 214, 92)
        strokeWidth = dp(1.4f)
        style = Paint.Style.STROKE
    }
    private val readyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(3f)
        strokeCap = Paint.Cap.ROUND
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    fun setState(newState: DirectorFrameState, mirrored: Boolean) {
        state = newState
        mirrorX = mirrored
        invalidate()
    }

    fun setPhotoPlan(newPlan: PhotoPlan?, newReadiness: PlanReadiness) {
        plan = newPlan
        readiness = newReadiness
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scene = state.scene ?: return
        val activePlan = plan

        if (state.gridEnabled || activePlan != null) {
            drawCompositionGuide(canvas, activePlan?.compositionGuide ?: CompositionGuide.THIRDS)
        }
        drawHorizon(canvas, scene.quality.deviceRollDeg)
        drawFocusTarget(canvas)
        drawReadiness(canvas)
    }

    private fun drawCompositionGuide(canvas: Canvas, guide: CompositionGuide) {
        when (guide) {
            CompositionGuide.NONE -> Unit
            CompositionGuide.THIRDS -> {
                val x1 = width / 3f
                val x2 = width * 2f / 3f
                val y1 = height / 3f
                val y2 = height * 2f / 3f
                canvas.drawLine(x1, 0f, x1, height.toFloat(), guidePaint)
                canvas.drawLine(x2, 0f, x2, height.toFloat(), guidePaint)
                canvas.drawLine(0f, y1, width.toFloat(), y1, guidePaint)
                canvas.drawLine(0f, y2, width.toFloat(), y2, guidePaint)
            }
            CompositionGuide.GOLDEN -> {
                val a = 0.382f
                val b = 0.618f
                canvas.drawLine(width * a, 0f, width * a, height.toFloat(), guidePaint)
                canvas.drawLine(width * b, 0f, width * b, height.toFloat(), guidePaint)
                canvas.drawLine(0f, height * a, width.toFloat(), height * a, guidePaint)
                canvas.drawLine(0f, height * b, width.toFloat(), height * b, guidePaint)
            }
            CompositionGuide.CENTER -> {
                val cx = width / 2f
                val cy = height / 2f
                canvas.drawLine(cx, height * 0.16f, cx, height * 0.84f, guidePaint)
                canvas.drawLine(width * 0.16f, cy, width * 0.84f, cy, guidePaint)
            }
            CompositionGuide.SYMMETRY -> {
                val cx = width / 2f
                canvas.drawLine(cx, 0f, cx, height.toFloat(), guidePaint)
            }
            CompositionGuide.DIAGONAL -> {
                canvas.drawLine(0f, height.toFloat(), width.toFloat(), 0f, guidePaint)
                canvas.drawLine(0f, 0f, width.toFloat(), height.toFloat(), guidePaint)
            }
        }
    }

    private fun drawHorizon(canvas: Canvas, rollDeg: Float) {
        if (abs(rollDeg) < 0.35f) return
        val cx = width / 2f
        val cy = height * 0.76f
        val half = dp(42f)
        val angle = Math.toRadians((-rollDeg).toDouble())
        val dx = (cos(angle) * half).toFloat()
        val dy = (sin(angle) * half).toFloat()
        horizonPaint.color = if (abs(rollDeg) <= 1.2f) Color.rgb(145, 255, 178) else Color.rgb(255, 210, 122)
        canvas.drawLine(cx - dx, cy - dy, cx + dx, cy + dy, horizonPaint)
        canvas.drawCircle(cx, cy, dp(2.6f), horizonPaint)
    }

    private fun drawFocusTarget(canvas: Canvas) {
        val p = plan ?: return
        val scene = state.scene ?: return
        if (!p.facePriority && p.focusIntent.name != "FACE" && p.focusIntent.name != "SUBJECT") return
        val pose = scene.poses.firstOrNull() ?: return
        val box = pose.bbox
        val left = if (mirrorX) 1f - box.right else box.left
        val right = if (mirrorX) 1f - box.left else box.right
        val rect = RectF(left * width, box.top * height, right * width, box.bottom * height)
        val insetX = rect.width() * 0.08f
        val insetY = rect.height() * 0.08f
        rect.inset(insetX, insetY)
        canvas.drawRoundRect(rect, dp(14f), dp(14f), focusPaint)
    }

    private fun drawReadiness(canvas: Canvas) {
        if (plan == null) return
        val x = width / 2f
        val y = height * 0.17f
        val r = dp(25f)
        readyPaint.color = Color.argb(90, 255, 255, 255)
        canvas.drawCircle(x, y, r, readyPaint)
        readyPaint.color = if (readiness.ready) Color.rgb(255, 210, 72) else Color.rgb(113, 247, 255)
        val sweep = 360f * readiness.score.coerceIn(0f, 1f)
        canvas.drawArc(RectF(x - r, y - r, x + r, y + r), -90f, sweep, false, readyPaint)
        textPaint.textSize = dp(8f)
        canvas.drawText(if (readiness.ready) "READY" else "AI", x, y + dp(3f), textPaint)
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
