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
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.saunak.director.ai.AiProviderPreset
import com.saunak.director.camera.CameraUiState
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorStage
import kotlinx.coroutines.delay
import kotlin.math.abs

private val Ink = Color(0xFF07090D)
private val Glass = Color(0xD512151C)
private val DirectorYellow = Color(0xFFFFD45C)
private val ReadyGreen = Color(0xFF98F5BE)
private val ErrorRed = Color(0xFFFF8A80)
private val Muted = Color(0xFFADB3C2)

@Composable
fun DirectorApp(vm: DirectorViewModel = viewModel()) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
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

    MaterialTheme(colorScheme = darkColorScheme(background = Ink, surface = Ink)) {
        if (!granted) PermissionScreen { launcher.launch(Manifest.permission.CAMERA) } else CameraScreen(vm)
    }
}

@Composable
private fun PermissionScreen(request: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Ink), contentAlignment = Alignment.Center) {
        Column(
            Modifier.padding(34.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(18.dp)
        ) {
            Box(Modifier.size(84.dp).border(1.dp, DirectorYellow.copy(alpha = .55f), CircleShape), contentAlignment = Alignment.Center) {
                Text("D", color = DirectorYellow, fontSize = 34.sp, fontWeight = FontWeight.Light)
            }
            Text("Director", color = Color.White, fontSize = 30.sp, fontWeight = FontWeight.SemiBold)
            Text(
                "Give Director camera access. AI provider keys stay encrypted on this device; only selected analysis frames are sent when you use Director.",
                color = Muted,
                textAlign = TextAlign.Center,
                lineHeight = 21.sp
            )
            Button(onClick = request, colors = ButtonDefaults.buttonColors(containerColor = DirectorYellow, contentColor = Color.Black)) {
                Text("Open camera", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun CameraScreen(vm: DirectorViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val haptics = LocalHapticFeedback.current
    val ui by vm.ui.collectAsStateWithLifecycle()
    var focusPoint by remember { mutableStateOf<Offset?>(null) }
    var showExposure by remember { mutableStateOf(false) }
    var requestedZoom by remember { mutableFloatStateOf(1f) }

    LaunchedEffect(ui.camera.zoom) { requestedZoom = ui.camera.zoom }
    LaunchedEffect(focusPoint) { if (focusPoint != null) { delay(900); focusPoint = null } }

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
                    detectTapGestures { point ->
                        focusPoint = point
                        haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        vm.focus(point.x, point.y)
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

        if (ui.gridEnabled) RuleOfThirdsGrid()
        if (ui.levelEnabled) LevelGuide()
        FocusRing(focusPoint)

        TopChrome(
            ui = ui,
            providerLabel = vm.selectedProvider().label,
            onSettings = vm::toggleSettings,
            onTools = vm::toggleTools,
            modifier = Modifier.align(Alignment.TopCenter)
        )

        if (showExposure && ui.camera.exposureSupported) {
            ExposureControl(ui.camera, Modifier.align(Alignment.CenterEnd).padding(end = 12.dp)) { vm.exposure(it) }
        }

        AnimatedVisibility(ui.showTools, Modifier.align(Alignment.CenterStart)) {
            ToolRail(
                ui.camera,
                showExposure,
                ui.gridEnabled,
                ui.levelEnabled,
                onTorch = { vm.torch(!ui.camera.torchEnabled) },
                onFlash = vm::cycleFlash,
                onExposure = { showExposure = !showExposure },
                onGrid = vm::toggleGrid,
                onLevel = vm::toggleLevel
            )
        }

        BottomChrome(
            ui = ui,
            requestedZoom = requestedZoom,
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
            onCapture = { haptics.performHapticFeedback(HapticFeedbackType.LongPress); vm.primaryCapture() },
            onSwitch = { haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove); vm.switchCamera() },
            modifier = Modifier.align(Alignment.BottomCenter)
        )

        if (ui.showSettings) ProviderSettingsSheet(vm, ui.recipeCount, vm::dismissSettings)
    }
}

@Composable
private fun TopChrome(
    ui: com.saunak.director.director.DirectorUiState,
    providerLabel: String,
    onSettings: () -> Unit,
    onTools: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 14.dp, vertical = 10.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            GlassButton("☰", onTools)
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("DIRECTOR", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Bold, letterSpacing = 3.sp)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Box(Modifier.size(6.dp).background(if (ui.hasApiKey) ReadyGreen else ErrorRed, CircleShape))
                    Text(providerLabel, color = Color.White.copy(alpha = .72f), fontSize = 9.sp, maxLines = 1)
                }
            }
            GlassButton("⚙", onSettings)
        }
        Spacer(Modifier.height(10.dp))
        StatusPill(ui, Modifier.align(Alignment.CenterHorizontally))
    }
}

@Composable
private fun StatusPill(ui: com.saunak.director.director.DirectorUiState, modifier: Modifier = Modifier) {
    val busy = ui.stage in setOf(
        DirectorStage.CAPTURING, DirectorStage.ANALYZING, DirectorStage.PLANNING,
        DirectorStage.APPLYING, DirectorStage.RECAPTURING, DirectorStage.CHECKING,
        DirectorStage.CORRECTING, DirectorStage.FINAL_CAPTURING, DirectorStage.SAVING
    )
    val tint = when (ui.stage) {
        DirectorStage.PREPARED -> DirectorYellow
        DirectorStage.ERROR -> ErrorRed
        else -> Color.White
    }
    Surface(color = Glass, shape = RoundedCornerShape(22.dp), modifier = modifier) {
        Column(Modifier.padding(horizontal = 14.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(9.dp)) {
                if (busy) CircularProgressIndicator(Modifier.size(13.dp), strokeWidth = 1.8.dp, color = DirectorYellow)
                else Box(Modifier.size(7.dp).background(tint, CircleShape))
                Text(ui.status, color = Color.White, fontSize = 12.sp, maxLines = 2)
                ui.lastScore?.let { Text("${(it * 100).toInt()}%", color = DirectorYellow, fontSize = 11.sp, fontWeight = FontWeight.Bold) }
            }
            if (ui.stage == DirectorStage.ERROR && !ui.error.isNullOrBlank()) {
                Text(ui.error.take(160), color = ErrorRed.copy(alpha = .9f), fontSize = 9.sp, maxLines = 2)
            }
        }
    }
}

@Composable
private fun BottomChrome(
    ui: com.saunak.director.director.DirectorUiState,
    requestedZoom: Float,
    onZoom: (Float) -> Unit,
    onGallery: () -> Unit,
    onCapture: () -> Unit,
    onSwitch: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier.fillMaxWidth().navigationBarsPadding().padding(bottom = 18.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(15.dp)
    ) {
        if (ui.recipeCount > 0) Surface(color = Glass, shape = RoundedCornerShape(18.dp)) {
            Text("✦ ${ui.recipeCount} learned recipe${if (ui.recipeCount == 1) "" else "s"}", Modifier.padding(horizontal = 11.dp, vertical = 5.dp), color = Color.White.copy(alpha = .75f), fontSize = 10.sp)
        }
        ZoomRow(ui.camera, requestedZoom, onZoom)
        Row(Modifier.fillMaxWidth().padding(horizontal = 30.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            GlassButton("▣", onGallery, 52.dp)
            DirectorShutter(ui.stage, ui.hasApiKey, onCapture)
            GlassButton("↻", onSwitch, 52.dp)
        }
        Text(
            if (ui.stage == DirectorStage.PREPARED) "READY • tap yellow shutter" else "Tap white shutter to let Director prepare the shot",
            color = if (ui.stage == DirectorStage.PREPARED) DirectorYellow else Color.White.copy(alpha = .62f),
            fontSize = 10.sp,
            letterSpacing = .5.sp
        )
    }
}

@Composable
private fun DirectorShutter(stage: DirectorStage, enabled: Boolean, onClick: () -> Unit) {
    val busy = stage in setOf(
        DirectorStage.CAPTURING, DirectorStage.ANALYZING, DirectorStage.PLANNING, DirectorStage.APPLYING,
        DirectorStage.RECAPTURING, DirectorStage.CHECKING, DirectorStage.CORRECTING,
        DirectorStage.FINAL_CAPTURING, DirectorStage.SAVING
    )
    val ready = stage == DirectorStage.PREPARED
    val ring = when { ready -> DirectorYellow; busy -> DirectorYellow.copy(alpha = .75f); else -> Color.White }
    val scale by animateFloatAsState(if (busy) .9f else 1f, label = "shutter")
    Surface(
        onClick = onClick,
        enabled = enabled && !busy,
        shape = CircleShape,
        color = Color.Transparent,
        border = androidx.compose.foundation.BorderStroke(3.dp, ring),
        modifier = Modifier.size((84 * scale).dp)
    ) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Box(Modifier.size(65.dp).background((if (ready) DirectorYellow else Color.White).copy(alpha = if (enabled) 1f else .4f), CircleShape))
            if (busy) CircularProgressIndicator(Modifier.size(50.dp), color = Ink, strokeWidth = 2.dp)
        }
    }
}

@Composable
private fun ProviderSettingsSheet(vm: DirectorViewModel, recipeCount: Int, dismiss: () -> Unit) {
    var selected by remember { mutableStateOf(vm.selectedProvider()) }
    var apiKey by remember(selected) { mutableStateOf("") }
    var endpoint by remember(selected) { mutableStateOf(if (selected == AiProviderPreset.CUSTOM_OPENAI) vm.customEndpoint() else "") }
    var model by remember(selected) {
        mutableStateOf(
            when (selected) {
                AiProviderPreset.CUSTOM_OPENAI -> vm.customModel()
                AiProviderPreset.OPENROUTER_CUSTOM -> vm.openRouterCustomModel()
                else -> selected.model.orEmpty()
            }
        )
    }
    var error by remember { mutableStateOf<String?>(null) }
    val scroll = rememberScrollState()

    Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = .66f)).padding(14.dp), contentAlignment = Alignment.Center) {
        Surface(color = Color(0xFA151820), shape = RoundedCornerShape(28.dp), tonalElevation = 10.dp, modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp).heightIn(max = 720.dp)) {
            Column(Modifier.padding(20.dp).verticalScroll(scroll), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column {
                        Text("AI Provider Hub", color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
                        Text("Switch cloud vision routes without rebuilding Director", color = Muted, fontSize = 11.sp)
                    }
                    TextButton(onClick = dismiss) { Text("✕", color = Color.White, fontSize = 20.sp) }
                }

                Text("Provider / model", color = Color.White, fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
                vm.providerPresets().forEach { preset ->
                    val active = selected == preset
                    val configured = preset == AiProviderPreset.AUTO && vm.providerPresets().any { it != AiProviderPreset.AUTO && vm.providerHasCredential(it) } ||
                        preset != AiProviderPreset.AUTO && vm.providerHasCredential(preset)
                    Surface(
                        onClick = {
                            selected = preset
                            error = null
                            vm.selectProvider(preset)
                        },
                        color = if (active) DirectorYellow.copy(alpha = .14f) else Color.White.copy(alpha = .05f),
                        border = androidx.compose.foundation.BorderStroke(1.dp, if (active) DirectorYellow else Color.White.copy(alpha = .08f)),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Row(Modifier.fillMaxWidth().padding(13.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                            Column(Modifier.weight(1f)) {
                                Text(preset.label, color = Color.White, fontSize = 12.sp, fontWeight = if (active) FontWeight.Bold else FontWeight.Medium)
                                Text(
                                    when {
                                        preset == AiProviderPreset.AUTO -> "Automatic failover across configured providers"
                                        preset.freeRoute -> "Free/trial-friendly route • provider limits apply"
                                        else -> "Direct selectable route"
                                    },
                                    color = Muted,
                                    fontSize = 9.sp
                                )
                            }
                            Box(Modifier.size(9.dp).background(if (configured) ReadyGreen else Color.White.copy(alpha = .22f), CircleShape))
                        }
                    }
                }

                if (selected != AiProviderPreset.AUTO) {
                    HorizontalDivider(color = Color.White.copy(alpha = .08f))
                    Text("Configure ${selected.label}", color = Color.White, fontWeight = FontWeight.SemiBold, fontSize = 12.sp)

                    if (selected == AiProviderPreset.CUSTOM_OPENAI) {
                        OutlinedTextField(
                            value = endpoint,
                            onValueChange = { endpoint = it; error = null },
                            label = { Text("HTTPS chat-completions endpoint") },
                            placeholder = { Text("https://host/v1/chat/completions") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    if (selected == AiProviderPreset.CUSTOM_OPENAI || selected == AiProviderPreset.OPENROUTER_CUSTOM) {
                        OutlinedTextField(
                            value = model,
                            onValueChange = { model = it; error = null },
                            label = { Text("Vision model ID") },
                            placeholder = { Text("provider/model-name") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    OutlinedTextField(
                        value = apiKey,
                        onValueChange = { apiKey = it; error = null },
                        label = { Text(if (vm.providerHasCredential(selected)) "Replace API key (optional)" else "API key") },
                        placeholder = { Text(providerKeyHint(selected)) },
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                    error?.let { Text(it, color = ErrorRed, fontSize = 11.sp) }

                    Button(
                        onClick = { error = vm.saveProvider(selected, apiKey, endpoint, model) },
                        enabled = apiKey.isNotBlank() || vm.providerHasCredential(selected),
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = DirectorYellow, contentColor = Color.Black)
                    ) {
                        Text("Save, test & use", fontWeight = FontWeight.Bold)
                    }

                    if (vm.providerHasCredential(selected)) {
                        TextButton(onClick = { vm.clearProvider(selected); apiKey = "" }, modifier = Modifier.fillMaxWidth()) {
                            Text("Remove ${selected.credentialId} key", color = ErrorRed)
                        }
                    }
                }

                HorizontalDivider(color = Color.White.copy(alpha = .08f))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    InfoMetric("Recipes", recipeCount.toString())
                    InfoMetric("Mode", vm.selectedProvider().label.substringBefore(" •"))
                    InfoMetric("Retries", "≤ 3")
                }
                Text(
                    "Auto priority: Groq Qwen → OpenRouter Nemotron Free → NVIDIA Direct → Gemini → Custom. Keys are encrypted separately with Android Keystore. Custom endpoints must use HTTPS.",
                    color = Muted,
                    fontSize = 10.sp,
                    lineHeight = 15.sp
                )
                Text(
                    "Privacy: free/trial hosted endpoints can have different data-use terms. Avoid sending sensitive/private faces or confidential imagery unless that provider's terms fit your use.",
                    color = DirectorYellow.copy(alpha = .82f),
                    fontSize = 9.sp,
                    lineHeight = 14.sp
                )
            }
        }
    }
}

private fun providerKeyHint(preset: AiProviderPreset): String = when (preset) {
    AiProviderPreset.NVIDIA_NEMOTRON -> "nvapi-…"
    AiProviderPreset.OPENROUTER_NEMOTRON_FREE, AiProviderPreset.OPENROUTER_NEMOTRON, AiProviderPreset.OPENROUTER_CUSTOM -> "sk-or-v1-…"
    AiProviderPreset.GROQ_QWEN38 -> "gsk_…"
    AiProviderPreset.GEMINI_38_FLASH -> "Google AI Studio key"
    AiProviderPreset.CUSTOM_OPENAI -> "Provider API key"
    AiProviderPreset.AUTO -> ""
}

@Composable
private fun ToolRail(
    camera: CameraUiState,
    showExposure: Boolean,
    grid: Boolean,
    level: Boolean,
    onTorch: () -> Unit,
    onFlash: () -> Unit,
    onExposure: () -> Unit,
    onGrid: () -> Unit,
    onLevel: () -> Unit
) {
    Surface(Modifier.padding(start = 12.dp), color = Glass, shape = RoundedCornerShape(26.dp)) {
        Column(Modifier.padding(7.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            ToolButton(if (camera.torchEnabled) "☀" else "☼", "Torch", camera.torchSupported, onTorch)
            val flash = when (camera.flashMode) { StillFlashMode.OFF -> "ϟ×"; StillFlashMode.AUTO -> "ϟA"; StillFlashMode.ON -> "ϟ" }
            ToolButton(flash, "Flash", camera.torchSupported, onFlash)
            ToolButton("±", "EV", camera.exposureSupported, onExposure, showExposure)
            ToolButton("#", "Grid", true, onGrid, grid)
            ToolButton("—", "Guide", true, onLevel, level)
        }
    }
}

@Composable
private fun ToolButton(symbol: String, label: String, enabled: Boolean, onClick: () -> Unit, active: Boolean = false) {
    TextButton(onClick = onClick, enabled = enabled, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 7.dp)) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(symbol, color = if (active) DirectorYellow else if (enabled) Color.White else Color.Gray, fontSize = 18.sp)
            Text(label, color = if (active) DirectorYellow else Color.White.copy(alpha = if (enabled) .6f else .25f), fontSize = 8.sp)
        }
    }
}

@Composable
private fun InfoMetric(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp)
        Text(label, color = Muted, fontSize = 9.sp)
    }
}

@Composable
private fun RuleOfThirdsGrid() {
    Canvas(Modifier.fillMaxSize()) {
        val c = Color.White.copy(alpha = .28f)
        drawLine(c, Offset(size.width / 3f, 0f), Offset(size.width / 3f, size.height), 1f)
        drawLine(c, Offset(size.width * 2f / 3f, 0f), Offset(size.width * 2f / 3f, size.height), 1f)
        drawLine(c, Offset(0f, size.height / 3f), Offset(size.width, size.height / 3f), 1f)
        drawLine(c, Offset(0f, size.height * 2f / 3f), Offset(size.width, size.height * 2f / 3f), 1f)
    }
}

@Composable
private fun LevelGuide() {
    Canvas(Modifier.fillMaxSize()) {
        val y = size.height / 2f
        val half = size.width * .11f
        drawLine(Color.White.copy(alpha = .65f), Offset(size.width / 2f - half, y), Offset(size.width / 2f + half, y), 3f, cap = StrokeCap.Round)
        drawCircle(DirectorYellow, 5f, Offset(size.width / 2f, y))
    }
}

@Composable
private fun FocusRing(point: Offset?) {
    Canvas(Modifier.fillMaxSize()) { point?.let { drawCircle(DirectorYellow, 30.dp.toPx(), it, style = Stroke(2.dp.toPx())) } }
}

@Composable
private fun ExposureControl(camera: CameraUiState, modifier: Modifier, onExposure: (Int) -> Unit) {
    Surface(modifier.width(58.dp).height(200.dp), color = Glass, shape = RoundedCornerShape(29.dp)) {
        Column(Modifier.padding(vertical = 7.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceBetween) {
            TextButton(onClick = { onExposure((camera.exposureIndex + 1).coerceAtMost(camera.exposureMaxIndex)) }) { Text("+", color = Color.White, fontSize = 22.sp) }
            val ev = camera.exposureIndex * camera.exposureStepEv
            Text(String.format(java.util.Locale.US, "%+.1f", ev), color = DirectorYellow, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            TextButton(onClick = { onExposure((camera.exposureIndex - 1).coerceAtLeast(camera.exposureMinIndex)) }) { Text("−", color = Color.White, fontSize = 22.sp) }
        }
    }
}

@Composable
private fun ZoomRow(camera: CameraUiState, current: Float, onZoom: (Float) -> Unit) {
    val options = remember(camera.minZoom, camera.maxZoom) {
        buildList {
            if (camera.minZoom < .95f) add(camera.minZoom)
            if (1f in camera.minZoom..camera.maxZoom) add(1f)
            if (2f in camera.minZoom..camera.maxZoom) add(2f)
            if (3f in camera.minZoom..camera.maxZoom) add(3f)
            if (5f in camera.minZoom..camera.maxZoom) add(5f)
            if (isEmpty()) add(camera.minZoom)
        }.distinctBy { (it * 100).toInt() }
    }
    Surface(color = Glass, shape = RoundedCornerShape(22.dp)) {
        Row(Modifier.padding(horizontal = 5.dp, vertical = 2.dp)) {
            options.forEach { value ->
                TextButton(onClick = { onZoom(value) }, contentPadding = PaddingValues(horizontal = 11.dp, vertical = 4.dp)) {
                    val label = if (abs(value - value.toInt()) < .02f) "${value.toInt()}×" else String.format(java.util.Locale.US, "%.1f×", value)
                    Text(label, color = if (abs(current - value) < .12f) DirectorYellow else Color.White, fontSize = 12.sp, fontWeight = if (abs(current - value) < .12f) FontWeight.Bold else FontWeight.Normal)
                }
            }
        }
    }
}

@Composable
private fun GlassButton(label: String, onClick: () -> Unit, size: androidx.compose.ui.unit.Dp = 44.dp) {
    Surface(onClick = onClick, shape = CircleShape, color = Glass, modifier = Modifier.size(size)) {
        Box(contentAlignment = Alignment.Center) { Text(label, color = Color.White, fontSize = 18.sp) }
    }
}
