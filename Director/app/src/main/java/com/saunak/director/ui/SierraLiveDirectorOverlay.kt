package com.saunak.director.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.PipelineEvent
import com.saunak.director.director.PipelineEventState
import com.saunak.director.director.live.DirectorGoalTypes
import com.saunak.director.director.live.DirectorRect
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
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
private val DirectorInk = Color(0xF2111318)

/** Preview visualization plus production observability surfaces. None of this UI can authorize capture. */
@Composable
internal fun SierraLiveDirectorOverlay(ui: DirectorUiState, modifier: Modifier = Modifier) {
    var showTrace by remember { mutableStateOf(false) }
    var showCompare by remember { mutableStateOf(false) }
    var dismissedSavedUri by remember { mutableStateOf<String?>(null) }
    val showCompletion = ui.showCompletion && ui.lastSavedUri != null && dismissedSavedUri != ui.lastSavedUri

    Box(modifier) {
        if (ui.liveDirector.running && ui.liveDirector.overlay.showTarget) {
            SierraDirectorCanvas(ui, Modifier.fillMaxSize())
        }

        if (ui.timeline.isNotEmpty() && ui.stage !in setOf(DirectorStage.CAMERA_ONLY, DirectorStage.READY)) {
            Surface(
                onClick = { showTrace = true },
                color = DirectorInk.copy(alpha = .88f),
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier.align(Alignment.CenterEnd).padding(end = 10.dp)
            ) {
                Column(Modifier.padding(horizontal = 9.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("TRACE", color = DirectorGold, fontSize = 8.sp, fontWeight = FontWeight.Bold)
                    Text(ui.timeline.size.toString(), color = Color.White, fontSize = 10.sp)
                }
            }
        }

        if (ui.stage in setOf(DirectorStage.FINAL_CAPTURING, DirectorStage.SAVING)) {
            ProcessingCard(ui, Modifier.align(Alignment.Center))
        }

        if (showTrace) PipelineTraceSheet(ui.timeline, onClose = { showTrace = false })
        if (showCompletion) {
            CompletionCard(
                ui = ui,
                showCompare = showCompare,
                onToggleCompare = { showCompare = !showCompare },
                onDismiss = { dismissedSavedUri = ui.lastSavedUri; showCompare = false }
            )
        }
    }
}

@Composable
private fun ProcessingCard(ui: DirectorUiState, modifier: Modifier = Modifier) {
    val recent = ui.timeline.takeLast(5)
    Surface(color = DirectorInk.copy(alpha = .94f), shape = RoundedCornerShape(22.dp), modifier = modifier.widthIn(max = 330.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(9.dp)) {
                CircularProgressIndicator(Modifier.size(16.dp), color = DirectorGold, strokeWidth = 1.8.dp)
                Text("Processing Sierra photo", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            }
            recent.forEach { event -> PipelineEventRow(event, compact = true) }
        }
    }
}

@Composable
private fun PipelineTraceSheet(events: List<PipelineEvent>, onClose: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .66f)).clickable { onClose() }.padding(18.dp), contentAlignment = Alignment.Center) {
        Surface(color = DirectorInk, shape = RoundedCornerShape(26.dp), modifier = Modifier.fillMaxWidth().widthIn(max = 560.dp).heightIn(max = 640.dp).clickable(enabled = false) {}) {
            Column(Modifier.padding(18.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column { Text("SHOT PIPELINE", color = Color.White, fontWeight = FontWeight.Bold, letterSpacing = 2.sp); Text("Real runtime events · no simulated progress", color = DirectorUnknown, fontSize = 10.sp) }
                    Text("✕", color = Color.White, modifier = Modifier.clickable { onClose() }.padding(8.dp))
                }
                HorizontalDivider(Modifier.padding(vertical = 12.dp), color = Color.White.copy(alpha = .08f))
                Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    events.forEach { PipelineEventRow(it, compact = false) }
                }
            }
        }
    }
}

@Composable
private fun PipelineEventRow(event: PipelineEvent, compact: Boolean) {
    val color = when (event.state) {
        PipelineEventState.COMPLETE -> DirectorReady
        PipelineEventState.CLAMPED -> DirectorGold
        PipelineEventState.REJECTED, PipelineEventState.FAILED -> Color(0xFFFF7373)
        PipelineEventState.ACTIVE -> DirectorGold
        PipelineEventState.INFO -> DirectorUnknown
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(9.dp), verticalAlignment = Alignment.Top) {
        Box(Modifier.padding(top = 5.dp).size(7.dp).background(color, CircleShape))
        Column(Modifier.weight(1f)) {
            Text(event.title, color = Color.White, fontSize = if (compact) 10.sp else 11.sp, fontWeight = FontWeight.Medium)
            if (!compact) {
                Text("${event.actor.name.replace('_',' ')} · ${event.stage.name.replace('_',' ')}", color = DirectorGold.copy(alpha = .82f), fontSize = 8.sp)
                event.detail?.takeIf(String::isNotBlank)?.let { Text(it, color = DirectorUnknown, fontSize = 9.sp, lineHeight = 13.sp) }
            }
        }
    }
}

@Composable
private fun CompletionCard(
    ui: DirectorUiState,
    showCompare: Boolean,
    onToggleCompare: () -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val finalUri = checkNotNull(ui.lastSavedUri)
    val finalBitmap = rememberFinalPreview(finalUri)
    val beforeBitmap = rememberFilePreview(ui.comparisonReferencePath)
    Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .70f)).padding(18.dp), contentAlignment = Alignment.Center) {
        Surface(color = DirectorInk, shape = RoundedCornerShape(28.dp), modifier = Modifier.fillMaxWidth().widthIn(max = 560.dp)) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("✓ SAVED TO GALLERY", color = DirectorReady, fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.5.sp)
                Text(if (showCompare) "Before Sierra render" else "Sierra final", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
                Surface(color = Color.Black, shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth().heightIn(min = 220.dp, max = 440.dp)) {
                    val bitmap = if (showCompare && beforeBitmap != null) beforeBitmap else finalBitmap
                    if (bitmap != null) Image(bitmap.asImageBitmap(), contentDescription = if (showCompare) "Before Sierra render" else "Sierra final photograph", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                    else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator(color = DirectorGold) }
                }
                Text(ui.finalReviewStatus ?: if (ui.sierraMode.name == "AI") "Sierra pre-authorized render" else "Sierra β measured render", color = DirectorUnknown, fontSize = 10.sp, textAlign = TextAlign.Center)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (beforeBitmap != null) TextButton(onClick = onToggleCompare) { Text(if (showCompare) "Show Sierra" else "Compare", color = DirectorGold) }
                    TextButton(onClick = {
                        runCatching { context.startActivity(Intent(Intent.ACTION_VIEW).apply { setDataAndType(Uri.parse(finalUri), "image/jpeg"); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) }) }
                    }) { Text("Open", color = Color.White) }
                    TextButton(onClick = {
                        runCatching { context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type = "image/jpeg"; putExtra(Intent.EXTRA_STREAM, Uri.parse(finalUri)); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) }, "Share Sierra photo")) }
                    }) { Text("Share", color = Color.White) }
                }
                Button(onClick = onDismiss, colors = ButtonDefaults.buttonColors(containerColor = DirectorGold, contentColor = Color.Black), shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
                    Text("Done", fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
private fun rememberFinalPreview(uriString: String): Bitmap? {
    val context = LocalContext.current
    val value by produceState<Bitmap?>(null, uriString) {
        value = withContext(Dispatchers.IO) { runCatching { context.contentResolver.loadThumbnail(Uri.parse(uriString), Size(1200, 1200), null) }.getOrNull() }
    }
    return value
}

@Composable
private fun rememberFilePreview(path: String?): Bitmap? {
    val value by produceState<Bitmap?>(null, path) {
        value = if (path.isNullOrBlank()) null else withContext(Dispatchers.IO) {
            runCatching { File(path).takeIf(File::isFile)?.let { BitmapFactory.decodeFile(it.absolutePath) } }.getOrNull()
        }
    }
    return value
}

@Composable
private fun SierraDirectorCanvas(ui: DirectorUiState, modifier: Modifier = Modifier) {
    val live = ui.liveDirector
    val overlay = live.overlay
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
            return Offset(offsetX + (nx.coerceIn(0.0, 1.0) * drawW).toFloat(), offsetY + (y.coerceIn(0.0, 1.0) * drawH).toFloat())
        }
        fun rect(value: DirectorRect): Rect {
            val a = point(value.left, value.top); val b = point(value.right, value.bottom)
            return Rect(minOf(a.x,b.x), minOf(a.y,b.y), maxOf(a.x,b.x), maxOf(a.y,b.y))
        }

        val confidenceAlpha = 0.30f + 0.70f * overlay.confidence.toFloat().coerceIn(0f, 1f)
        overlay.targetBox?.let { target ->
            val r = rect(target)
            drawRect(if (live.captureReady) DirectorReady else DirectorGold.copy(alpha = confidenceAlpha), r.topLeft, r.size,
                style = Stroke(width = if (live.captureReady) 4f else 3f, pathEffect = if (live.captureReady) null else PathEffect.dashPathEffect(floatArrayOf(18f, 12f), 0f)))
            drawTargetCorners(r, if (live.captureReady) DirectorReady else DirectorGold)
        }
        overlay.currentBox?.let { current ->
            val r = rect(current)
            val c = when { live.captureReady -> DirectorReady; live.entityLocked -> DirectorCurrent.copy(alpha = confidenceAlpha); else -> DirectorUnknown.copy(alpha=.65f) }
            drawRect(c, r.topLeft, r.size, style = Stroke(2.5f)); drawCircle(c, 4f, r.center)
        }
        overlay.targetKeypoints.forEach { kp -> val p = point(kp.x,kp.y); drawCircle(DirectorGold.copy(alpha=.72f),9f,p,style=Stroke(2f)); drawCircle(DirectorGold.copy(alpha=.32f),3f,p) }
        overlay.currentKeypoints.forEach { kp -> drawCircle(DirectorCurrent.copy(alpha=confidenceAlpha),4.5f,point(kp.x,kp.y)) }

        live.instruction?.takeIf { !it.achieved }?.let { instruction ->
            val dx = instruction.directionX; val dy = instruction.directionY; val magnitude = sqrt(dx*dx+dy*dy)
            if (magnitude > 0.008) {
                val visualDx = if (mirror) -dx else dx; val norm = sqrt(visualDx*visualDx+dy*dy).coerceAtLeast(1e-6)
                val origin = overlay.currentBox?.let { rect(it).center } ?: Offset(size.width*.5f,size.height*.5f)
                val length = 70f + instruction.urgency.toFloat().coerceIn(0f,1f)*55f
                drawArrow(origin, Offset(origin.x+(visualDx/norm).toFloat()*length, origin.y+(dy/norm).toFloat()*length), DirectorGold)
            }
        }
        if (overlay.activeGoalType == DirectorGoalTypes.LIGHT_BALANCE_X) {
            val evaluation = live.evaluations.firstOrNull { it.node.type == DirectorGoalTypes.LIGHT_BALANCE_X }
            if (evaluation != null) {
                val target = when { evaluation.node.targetMin != null && evaluation.node.targetMax != null -> (evaluation.node.targetMin + evaluation.node.targetMax) * .5; evaluation.node.targetMin != null -> evaluation.node.targetMin; evaluation.node.targetMax != null -> evaluation.node.targetMax; else -> 0.0 }
                val box = overlay.currentBox?.let(::rect) ?: overlay.targetBox?.let(::rect)
                if (box != null && abs(target) > .04) {
                    val visualRight = if (mirror) target <= 0.0 else target > 0.0
                    val x = if (visualRight) box.right+24f else box.left-24f
                    val marker = Offset(x.coerceIn(20f,size.width-20f), box.center.y.coerceIn(20f,size.height-20f))
                    drawCircle(DirectorGold.copy(alpha=.25f),22f,marker); drawCircle(DirectorGold,8f,marker); drawLine(DirectorGold,marker,box.center,2f,StrokeCap.Round)
                }
            }
        }
        if (live.captureReady) {
            val c=Offset(size.width*.5f,size.height*.5f); drawCircle(DirectorReady.copy(alpha=.12f),50f,c); drawCircle(DirectorReady.copy(alpha=.78f),18f,c,style=Stroke(3f))
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawTargetCorners(rect: Rect, color: Color) {
    val length = minOf(rect.width, rect.height).coerceAtMost(42f).coerceAtLeast(18f); val w=4f
    drawLine(color,rect.topLeft,Offset(rect.left+length,rect.top),w,StrokeCap.Round); drawLine(color,rect.topLeft,Offset(rect.left,rect.top+length),w,StrokeCap.Round)
    drawLine(color,rect.topRight,Offset(rect.right-length,rect.top),w,StrokeCap.Round); drawLine(color,rect.topRight,Offset(rect.right,rect.top+length),w,StrokeCap.Round)
    drawLine(color,rect.bottomLeft,Offset(rect.left+length,rect.bottom),w,StrokeCap.Round); drawLine(color,rect.bottomLeft,Offset(rect.left,rect.bottom-length),w,StrokeCap.Round)
    drawLine(color,rect.bottomRight,Offset(rect.right-length,rect.bottom),w,StrokeCap.Round); drawLine(color,rect.bottomRight,Offset(rect.right,rect.bottom-length),w,StrokeCap.Round)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawArrow(start: Offset, end: Offset, color: Color) {
    drawLine(color,start,end,5f,StrokeCap.Round)
    val angle=atan2((end.y-start.y).toDouble(),(end.x-start.x).toDouble()); val head=20.0; val wing=.62
    val a=Offset((end.x-head*cos(angle-wing)).toFloat(),(end.y-head*sin(angle-wing)).toFloat()); val b=Offset((end.x-head*cos(angle+wing)).toFloat(),(end.y-head*sin(angle+wing)).toFloat())
    drawLine(color,end,a,5f,StrokeCap.Round); drawLine(color,end,b,5f,StrokeCap.Round)
}
