package com.saunak.director.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.SierraMode
import kotlinx.coroutines.delay
import kotlin.math.min

private val SierraInk = Color(0xFF090A0C)
private val SierraGraphite = Color(0xF2111318)
private val SierraGold = Color(0xFFF2B84B)
private val SierraSoftGold = Color(0xFFFFD77A)
private val SierraIvory = Color(0xFFF5F6F8)
private val SierraSilver = Color(0xFFA4ABB5)
private val SierraSuccess = Color(0xFF6FD99C)
private val SierraError = Color(0xFFFF5D5D)

@Composable
fun SierraVisionApp(vm: DirectorViewModel = viewModel()) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    var explainedPermission by remember { mutableStateOf(granted) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val lifecycleOwner = LocalLifecycleOwner.current

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> {
                    granted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                    vm.onAppForegrounded()
                }
                Lifecycle.Event.ON_STOP -> vm.onAppBackgrounded()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    MaterialTheme(colorScheme = darkColorScheme(background = SierraInk, surface = SierraGraphite, primary = SierraGold)) {
        when {
            !explainedPermission -> SierraPermissionIntro { explainedPermission = true; launcher.launch(Manifest.permission.CAMERA) }
            !granted -> SierraPermissionDenied { launcher.launch(Manifest.permission.CAMERA) }
            else -> SierraCameraScreen(vm)
        }
    }
}

@Composable
private fun SierraPermissionIntro(request: () -> Unit) {
    Box(Modifier.fillMaxSize().background(SierraInk).padding(32.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(18.dp)) {
            SierraMark(88.dp)
            Text("SIERRA VISION", color = SierraIvory, fontSize = 25.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 4.sp)
            Text("Photography, directed by intelligence.", color = SierraSilver, fontSize = 15.sp)
            Spacer(Modifier.height(8.dp))
            Text(
                "Sierra Vision needs camera access to frame and capture photographs. Sierra β works on-device; only a reduced analysis frame may leave the phone when Sierra AI is used.",
                color = SierraSilver,
                textAlign = TextAlign.Center,
                lineHeight = 21.sp
            )
            Button(onClick = request, colors = ButtonDefaults.buttonColors(containerColor = SierraGold, contentColor = Color.Black), shape = RoundedCornerShape(18.dp)) {
                Text("Open camera", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun SierraPermissionDenied(request: () -> Unit) {
    Box(Modifier.fillMaxSize().background(SierraInk).padding(32.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
            SierraMark(72.dp)
            Text("Camera access is required", color = SierraIvory, fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
            Text("Sierra Vision can’t photograph without Android camera permission.", color = SierraSilver, textAlign = TextAlign.Center)
            Button(onClick = request, colors = ButtonDefaults.buttonColors(containerColor = SierraGold, contentColor = Color.Black)) { Text("Allow camera") }
        }
    }
}

@Composable
private fun SierraCameraScreen(vm: DirectorViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val haptics = LocalHapticFeedback.current
    val ui by vm.ui.collectAsStateWithLifecycle()
    var focusPoint by remember { mutableStateOf<Offset?>(null) }
    var requestedZoom by remember { mutableFloatStateOf(1f) }
    var showStatus by remember { mutableStateOf(false) }

    LaunchedEffect(ui.camera.zoom) { requestedZoom = ui.camera.zoom }
    LaunchedEffect(focusPoint) { if (focusPoint != null) { delay(850); focusPoint = null } }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                    vm.bind(lifecycleOwner, this)
                }
            },
            modifier = Modifier.fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures { p ->
                        focusPoint = p
                        haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        vm.focus(p.x, p.y)
                    }
                }
                .pointerInput(ui.camera.minZoom, ui.camera.maxZoom) {
                    detectTransformGestures { _, _, gestureZoom, _ ->
                        if (gestureZoom != 1f) {
                            requestedZoom = (requestedZoom * gestureZoom).coerceIn(ui.camera.minZoom, ui.camera.maxZoom)
                            vm.zoom(requestedZoom)
                        }
                    }
                }
        )

        if (ui.gridEnabled) SierraRuleOfThirds()
        if (ui.levelEnabled) SierraLevel()
        SierraCompositionFrame(ui)
        SierraFocusRing(focusPoint)

        SierraTopBar(ui, onSierra = { showStatus = true }, onTools = vm::toggleTools, modifier = Modifier.align(Alignment.TopCenter))
        SierraGuidance(ui, Modifier.align(Alignment.TopCenter).padding(top = 105.dp))

        SierraBottomBar(
            ui = ui,
            zoom = requestedZoom,
            onZoom = { requestedZoom = it; vm.zoom(it) },
            onGallery = {
                val intent = ui.lastSavedUri?.let { value ->
                    Intent(Intent.ACTION_VIEW).apply {
                        setDataAndType(Uri.parse(value), "image/jpeg")
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                } ?: Intent(Intent.ACTION_VIEW, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                runCatching { context.startActivity(intent) }
            },
            onCapture = {
                haptics.performHapticFeedback(if (ui.stage == DirectorStage.PREPARED) HapticFeedbackType.LongPress else HapticFeedbackType.TextHandleMove)
                vm.primaryCapture()
            },
            onSwitch = vm::switchCamera,
            modifier = Modifier.align(Alignment.BottomCenter)
        )

        AnimatedVisibility(ui.showTools, Modifier.align(Alignment.BottomCenter)) {
            SierraToolsSheet(ui, vm, onClose = vm::dismissTools)
        }
        if (showStatus) SierraStatusSheet(ui, vm, onClose = { showStatus = false })
    }
}

@Composable
private fun SierraTopBar(ui: DirectorUiState, onSierra: () -> Unit, onTools: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Surface(onClick = onSierra, color = SierraGraphite.copy(alpha = .78f), shape = RoundedCornerShape(18.dp)) {
            Row(Modifier.padding(horizontal = 13.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("SIERRA", color = SierraIvory, fontWeight = FontWeight.Bold, letterSpacing = 2.sp, fontSize = 13.sp)
                Box(Modifier.size(7.dp).background(if (ui.sierraMode == SierraMode.AI) SierraSuccess else SierraSilver, CircleShape))
                Text(if (ui.sierraMode == SierraMode.AI) "AI" else "β", color = if (ui.sierraMode == SierraMode.AI) SierraSuccess else SierraSilver, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
        }
        Surface(onClick = onTools, color = SierraGraphite.copy(alpha = .78f), shape = CircleShape, modifier = Modifier.size(44.dp)) {
            Box(contentAlignment = Alignment.Center) { Text("•••", color = SierraIvory, fontSize = 16.sp, letterSpacing = 1.sp) }
        }
    }
}

@Composable
private fun SierraGuidance(ui: DirectorUiState, modifier: Modifier = Modifier) {
    val visible = ui.stage !in setOf(DirectorStage.READY, DirectorStage.CAMERA_ONLY) || ui.error != null
    AnimatedVisibility(visible, modifier) {
        Surface(color = SierraGraphite.copy(alpha = .84f), shape = RoundedCornerShape(22.dp)) {
            Row(Modifier.padding(horizontal = 14.dp, vertical = 9.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(9.dp)) {
                if (ui.stage in setOf(DirectorStage.ANALYZING, DirectorStage.PLANNING, DirectorStage.APPLYING, DirectorStage.CHECKING, DirectorStage.SAVING, DirectorStage.FINAL_CAPTURING)) {
                    CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 1.7.dp, color = SierraGold)
                } else {
                    Box(Modifier.size(7.dp).background(if (ui.stage == DirectorStage.ERROR) SierraError else SierraGold, CircleShape))
                }
                Column {
                    Text(ui.status, color = SierraIvory, fontSize = 12.sp, maxLines = 2)
                    if (ui.stage == DirectorStage.PREPARED) {
                        Text(listOfNotNull(ui.sceneLabel?.replace('_', ' ')?.replaceFirstChar { it.uppercase() }, ui.aspectRatio).joinToString(" · "), color = SierraSoftGold, fontSize = 10.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun SierraBottomBar(
    ui: DirectorUiState,
    zoom: Float,
    onZoom: (Float) -> Unit,
    onGallery: () -> Unit,
    onCapture: () -> Unit,
    onSwitch: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier.fillMaxWidth().navigationBarsPadding().padding(bottom = 18.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(13.dp)
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(9.dp), verticalAlignment = Alignment.CenterVertically) {
            listOf(0.5f, 1f, 2f).filter { it >= ui.camera.minZoom && it <= ui.camera.maxZoom }.forEach { value ->
                val active = kotlin.math.abs(zoom - value) < .18f
                Surface(onClick = { onZoom(value) }, color = if (active) SierraIvory else SierraGraphite.copy(alpha = .76f), contentColor = if (active) Color.Black else SierraIvory, shape = CircleShape, modifier = Modifier.size(if (active) 42.dp else 36.dp)) {
                    Box(contentAlignment = Alignment.Center) { Text(if (value == 1f) "1×" else "${value}×", fontSize = 11.sp, fontWeight = FontWeight.Bold) }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(horizontal = 30.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            SierraRoundAction("▣", onGallery)
            SierraShutter(ui, onCapture)
            SierraRoundAction("↻", onSwitch)
        }
        Text(
            when (ui.stage) {
                DirectorStage.PREPARED -> "READY • tap gold shutter"
                DirectorStage.SAVING -> "Finishing your Sierra photo"
                else -> if (ui.privateMode) "Sierra β Private • no cloud" else "Tap white shutter to prepare the shot"
            },
            color = if (ui.stage == DirectorStage.PREPARED) SierraSoftGold else SierraIvory.copy(alpha = .66f),
            fontSize = 10.sp
        )
    }
}

@Composable
private fun SierraShutter(ui: DirectorUiState, onClick: () -> Unit) {
    val busy = ui.stage in setOf(DirectorStage.CAPTURING, DirectorStage.ANALYZING, DirectorStage.PLANNING, DirectorStage.APPLYING, DirectorStage.CHECKING, DirectorStage.FINAL_CAPTURING, DirectorStage.SAVING)
    val ready = ui.stage == DirectorStage.PREPARED
    val scale by animateFloatAsState(if (busy) .92f else 1f, label = "sierraShutter")
    val ring = if (ready) SierraGold else SierraIvory
    Surface(onClick = onClick, enabled = !busy, color = Color.Transparent, border = androidx.compose.foundation.BorderStroke(3.dp, ring), shape = CircleShape, modifier = Modifier.size((86 * scale).dp)) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Box(Modifier.size(66.dp).background(if (ready) SierraGold else SierraIvory, CircleShape))
            if (busy) CircularProgressIndicator(Modifier.size(48.dp), color = SierraInk, strokeWidth = 2.dp)
        }
    }
}

@Composable
private fun SierraToolsSheet(ui: DirectorUiState, vm: DirectorViewModel, onClose: () -> Unit) {
    Surface(color = SierraGraphite, shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.navigationBarsPadding().padding(20.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text("Camera Tools", color = SierraIvory, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                Text("✕", color = SierraIvory, modifier = Modifier.clickable { onClose() }.padding(8.dp))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                SierraTool("Flash", ui.camera.flashMode.wireName.replaceFirstChar { it.uppercase() }) { vm.cycleFlash() }
                SierraTool("Grid", if (ui.gridEnabled) "On" else "Off") { vm.toggleGrid() }
                SierraTool("Level", if (ui.levelEnabled) "On" else "Off") { vm.toggleLevel() }
                SierraTool("Ratio", ui.aspectRatio) { }
            }
            HorizontalDivider(color = Color.White.copy(alpha = .08f))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Column {
                    Text("Private Photography", color = SierraIvory, fontWeight = FontWeight.SemiBold)
                    Text("Sierra β only · no image leaves device", color = SierraSilver, fontSize = 11.sp)
                }
                Switch(checked = ui.privateMode, onCheckedChange = { vm.togglePrivateMode() }, colors = SwitchDefaults.colors(checkedThumbColor = Color.Black, checkedTrackColor = SierraGold))
            }
            Text("Final photos save to Pictures/Sierra Vision. Sierra β remains available even when Sierra AI is unavailable.", color = SierraSilver, fontSize = 11.sp, lineHeight = 16.sp)
        }
    }
}

@Composable
private fun SierraStatusSheet(ui: DirectorUiState, vm: DirectorViewModel, onClose: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .62f)).clickable { onClose() }.padding(18.dp), contentAlignment = Alignment.Center) {
        Surface(color = SierraGraphite, shape = RoundedCornerShape(28.dp), modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp).clickable(enabled = false) { }) {
            Column(Modifier.padding(22.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("SIERRA", color = SierraIvory, letterSpacing = 3.sp, fontWeight = FontWeight.Bold)
                    Text("✕", color = SierraIvory, modifier = Modifier.clickable { onClose() }.padding(8.dp))
                }
                SierraStatusRow("Sierra AI", if (ui.aiAvailable && !ui.privateMode) "Available" else if (ui.privateMode) "Disabled by Private Photography" else "Unavailable")
                SierraStatusRow("Sierra β", "Active")
                SierraStatusRow("Current intelligence", if (ui.sierraMode == SierraMode.AI) "Sierra AI + Sierra β" else "Sierra β")
                SierraStatusRow("Composition", ui.aspectRatio)
                HorizontalDivider(color = Color.White.copy(alpha = .08f))
                Text("Sierra AI directs the photograph when available. Sierra β always validates and executes camera controls locally, and takes over planning automatically when cloud intelligence is unavailable.", color = SierraSilver, fontSize = 12.sp, lineHeight = 18.sp)
                if (ui.privateMode) Text("Private Photography is active. No cloud analysis will be attempted.", color = SierraSoftGold, fontSize = 12.sp)
                TextButton(onClick = { vm.togglePrivateMode() }) { Text(if (ui.privateMode) "Disable Private Photography" else "Use Sierra β privately", color = SierraGold) }
            }
        }
    }
}

@Composable
private fun SierraStatusRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = SierraSilver, fontSize = 12.sp)
        Text(value, color = SierraIvory, fontSize = 12.sp, fontWeight = FontWeight.Medium, textAlign = TextAlign.End)
    }
}

@Composable
private fun SierraTool(label: String, value: String, onClick: () -> Unit) {
    Surface(onClick = onClick, color = Color.White.copy(alpha = .05f), shape = RoundedCornerShape(16.dp), modifier = Modifier.width(76.dp)) {
        Column(Modifier.padding(vertical = 12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(label, color = SierraIvory, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
            Text(value, color = SierraSilver, fontSize = 9.sp, maxLines = 1)
        }
    }
}

@Composable
private fun SierraRoundAction(text: String, onClick: () -> Unit) {
    Surface(onClick = onClick, color = SierraGraphite.copy(alpha = .80f), shape = CircleShape, modifier = Modifier.size(52.dp)) {
        Box(contentAlignment = Alignment.Center) { Text(text, color = SierraIvory, fontSize = 20.sp) }
    }
}

@Composable
private fun SierraCompositionFrame(ui: DirectorUiState) {
    if (ui.stage !in setOf(DirectorStage.APPLYING, DirectorStage.CHECKING, DirectorStage.PREPARED, DirectorStage.FINAL_CAPTURING)) return
    Canvas(Modifier.fillMaxSize()) {
        val ratio = parseRatio(ui.aspectRatio) ?: return@Canvas
        val availableW = size.width * .90f
        val availableH = size.height * .70f
        val target = ratio.first / ratio.second
        var w = availableW
        var h = w / target
        if (h > availableH) { h = availableH; w = h * target }
        val left = (size.width - w) / 2f
        val top = (size.height - h) / 2f
        val rect = Rect(left, top, left + w, top + h)
        drawRect(Color.Black.copy(alpha = .26f), topLeft = Offset(0f, 0f), size = androidx.compose.ui.geometry.Size(size.width, rect.top))
        drawRect(Color.Black.copy(alpha = .26f), topLeft = Offset(0f, rect.bottom), size = androidx.compose.ui.geometry.Size(size.width, size.height - rect.bottom))
        drawRect(Color.Black.copy(alpha = .26f), topLeft = Offset(0f, rect.top), size = androidx.compose.ui.geometry.Size(rect.left, rect.height))
        drawRect(Color.Black.copy(alpha = .26f), topLeft = Offset(rect.right, rect.top), size = androidx.compose.ui.geometry.Size(size.width - rect.right, rect.height))
        drawRect(if (ui.stage == DirectorStage.PREPARED) SierraGold else SierraIvory.copy(alpha = .65f), topLeft = rect.topLeft, size = rect.size, style = Stroke(width = if (ui.stage == DirectorStage.PREPARED) 3f else 2f))
    }
}

private fun parseRatio(value: String): Pair<Float, Float>? = when (value) {
    "4:3" -> 4f to 3f; "3:2" -> 3f to 2f; "4:5" -> 4f to 5f; "1:1" -> 1f to 1f
    "16:9" -> 16f to 9f; "9:16" -> 9f to 16f; "2.39:1" -> 2.39f to 1f
    else -> null
}

@Composable
private fun SierraRuleOfThirds() = Canvas(Modifier.fillMaxSize()) {
    val c = Color.White.copy(alpha = .23f)
    for (f in listOf(1f / 3f, 2f / 3f)) {
        drawLine(c, Offset(size.width * f, 0f), Offset(size.width * f, size.height), 1f)
        drawLine(c, Offset(0f, size.height * f), Offset(size.width, size.height * f), 1f)
    }
}

@Composable
private fun SierraLevel() = Canvas(Modifier.fillMaxSize()) {
    val y = size.height * .5f
    val w = min(size.width * .2f, 150f)
    drawLine(SierraGold.copy(alpha = .85f), Offset(size.width / 2f - w, y), Offset(size.width / 2f + w, y), 3f, StrokeCap.Round)
}

@Composable
private fun SierraFocusRing(point: Offset?) {
    if (point == null) return
    Canvas(Modifier.fillMaxSize()) {
        drawCircle(SierraGold, 38f, point, style = Stroke(2.5f))
        drawCircle(SierraGold, 3.5f, point)
    }
}

@Composable
private fun SierraMark(size: androidx.compose.ui.unit.Dp) {
    Box(Modifier.size(size).border(2.dp, SierraGold, CircleShape), contentAlignment = Alignment.Center) {
        Text("S", color = SierraGold, fontSize = (size.value * .42f).sp, fontWeight = FontWeight.Light)
    }
}
