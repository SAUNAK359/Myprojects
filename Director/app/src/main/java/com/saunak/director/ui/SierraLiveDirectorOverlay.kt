package com.saunak.director.ui

import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.live.DirectorGoalTypes
import com.saunak.director.director.live.DirectorRect
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

private val DirectorGold = Color(0xFFF2B84B)
private val DirectorCurrent = Color(0xFFE8ECF2)
private val DirectorReady = Color(0xFF6FD99C)
private val DirectorUnknown = Color(0xFF9AA3AF)

/**
 * Preview-only visualization. It never affects readiness. The runtime is the authority; this layer
 * simply projects its normalized oriented-analysis coordinates over PreviewView.FILL_CENTER.
 */
@Composable
internal fun SierraLiveDirectorOverlay(ui: DirectorUiState, modifier: Modifier = Modifier) {
    val live = ui.liveDirector
    val overlay = live.overlay
    if (!live.running || !overlay.showTarget) return

    Canvas(modifier) {
        val sourceW = overlay.sourceWidth.takeIf { it > 0 } ?: 4
        val sourceH = overlay.sourceHeight.takeIf { it > 0 } ?: 3
        val scale = max(size.width / sourceW.toFloat(), size.height / sourceH.toFloat())
        val drawW = sourceW * scale
        val drawH = sourceH * scale
        val offsetX = (size.width - drawW) * 0.5f
        val offsetY = (size.height - drawH) * 0.5f
        val mirror = ui.camera.lensFacing == CameraSelector.LENS_FACING_FRONT

        fun point(x: Double, y: Double): Offset {
            val nx = if (mirror) 1.0 - x else x
            return Offset(
                offsetX + (nx.coerceIn(0.0, 1.0) * drawW).toFloat(),
                offsetY + (y.coerceIn(0.0, 1.0) * drawH).toFloat()
            )
        }
        fun rect(value: DirectorRect): Rect {
            val a = point(value.left, value.top)
            val b = point(value.right, value.bottom)
            return Rect(
                left = minOf(a.x, b.x),
                top = minOf(a.y, b.y),
                right = maxOf(a.x, b.x),
                bottom = maxOf(a.y, b.y)
            )
        }

        val confidenceAlpha = 0.30f + 0.70f * overlay.confidence.toFloat().coerceIn(0f, 1f)
        overlay.targetBox?.let { target ->
            val r = rect(target)
            drawRect(
                color = if (live.captureReady) DirectorReady else DirectorGold.copy(alpha = confidenceAlpha),
                topLeft = r.topLeft,
                size = r.size,
                style = Stroke(
                    width = if (live.captureReady) 4f else 3f,
                    pathEffect = if (live.captureReady) null else PathEffect.dashPathEffect(floatArrayOf(18f, 12f), 0f)
                )
            )
            drawTargetCorners(r, if (live.captureReady) DirectorReady else DirectorGold)
        }

        overlay.currentBox?.let { current ->
            val r = rect(current)
            val currentColor = when {
                live.captureReady -> DirectorReady
                live.entityLocked -> DirectorCurrent.copy(alpha = confidenceAlpha)
                else -> DirectorUnknown.copy(alpha = .65f)
            }
            drawRect(currentColor, r.topLeft, r.size, style = Stroke(width = 2.5f))
            drawCircle(currentColor, radius = 4f, center = r.center)
        }

        overlay.targetKeypoints.forEach { kp ->
            val p = point(kp.x, kp.y)
            drawCircle(DirectorGold.copy(alpha = .72f), 9f, p, style = Stroke(width = 2f))
            drawCircle(DirectorGold.copy(alpha = .32f), 3f, p)
        }
        overlay.currentKeypoints.forEach { kp ->
            drawCircle(DirectorCurrent.copy(alpha = confidenceAlpha), 4.5f, point(kp.x, kp.y))
        }

        val instruction = live.instruction
        if (instruction != null && !instruction.achieved) {
            val dx = instruction.directionX
            val dy = instruction.directionY
            val magnitude = sqrt(dx * dx + dy * dy)
            if (magnitude > 0.008) {
                val visualDx = if (mirror) -dx else dx
                val norm = sqrt(visualDx * visualDx + dy * dy).coerceAtLeast(1e-6)
                val origin = overlay.currentBox?.let { rect(it).center } ?: Offset(size.width * 0.5f, size.height * 0.5f)
                val length = 70f + instruction.urgency.toFloat().coerceIn(0f, 1f) * 55f
                val end = Offset(
                    origin.x + (visualDx / norm).toFloat() * length,
                    origin.y + (dy / norm).toFloat() * length
                )
                drawArrow(origin, end, DirectorGold)
            }
        }

        if (overlay.activeGoalType == DirectorGoalTypes.LIGHT_BALANCE_X) {
            val evaluation = live.evaluations.firstOrNull { it.node.type == DirectorGoalTypes.LIGHT_BALANCE_X }
            if (evaluation != null) {
                val min = evaluation.node.targetMin
                val maxValue = evaluation.node.targetMax
                val target = when {
                    min != null && maxValue != null -> (min + maxValue) * 0.5
                    min != null -> min
                    maxValue != null -> maxValue
                    else -> 0.0
                }
                val box = overlay.currentBox?.let(::rect) ?: overlay.targetBox?.let(::rect)
                if (box != null && abs(target) > 0.04) {
                    val desiredRight = target > 0.0
                    val visualRight = if (mirror) !desiredRight else desiredRight
                    val x = if (visualRight) box.right + 24f else box.left - 24f
                    val marker = Offset(x.coerceIn(20f, size.width - 20f), box.center.y.coerceIn(20f, size.height - 20f))
                    drawCircle(DirectorGold.copy(alpha = .25f), 22f, marker)
                    drawCircle(DirectorGold, 8f, marker)
                    drawLine(DirectorGold, marker, box.center, 2f, StrokeCap.Round)
                }
            }
        }

        if (live.captureReady) {
            val c = Offset(size.width * 0.5f, size.height * 0.5f)
            drawCircle(DirectorReady.copy(alpha = .12f), radius = 50f, center = c)
            drawCircle(DirectorReady.copy(alpha = .78f), radius = 18f, center = c, style = Stroke(width = 3f))
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawTargetCorners(rect: Rect, color: Color) {
    val length = minOf(rect.width, rect.height).coerceAtMost(42f).coerceAtLeast(18f)
    val w = 4f
    drawLine(color, rect.topLeft, Offset(rect.left + length, rect.top), w, StrokeCap.Round)
    drawLine(color, rect.topLeft, Offset(rect.left, rect.top + length), w, StrokeCap.Round)
    drawLine(color, rect.topRight, Offset(rect.right - length, rect.top), w, StrokeCap.Round)
    drawLine(color, rect.topRight, Offset(rect.right, rect.top + length), w, StrokeCap.Round)
    drawLine(color, rect.bottomLeft, Offset(rect.left + length, rect.bottom), w, StrokeCap.Round)
    drawLine(color, rect.bottomLeft, Offset(rect.left, rect.bottom - length), w, StrokeCap.Round)
    drawLine(color, rect.bottomRight, Offset(rect.right - length, rect.bottom), w, StrokeCap.Round)
    drawLine(color, rect.bottomRight, Offset(rect.right, rect.bottom - length), w, StrokeCap.Round)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawArrow(start: Offset, end: Offset, color: Color) {
    drawLine(color, start, end, 5f, StrokeCap.Round)
    val angle = atan2((end.y - start.y).toDouble(), (end.x - start.x).toDouble())
    val head = 20.0
    val wing = 0.62
    val a = Offset(
        (end.x - head * cos(angle - wing)).toFloat(),
        (end.y - head * sin(angle - wing)).toFloat()
    )
    val b = Offset(
        (end.x - head * cos(angle + wing)).toFloat(),
        (end.y - head * sin(angle + wing)).toFloat()
    )
    drawLine(color, end, a, 5f, StrokeCap.Round)
    drawLine(color, end, b, 5f, StrokeCap.Round)
}
