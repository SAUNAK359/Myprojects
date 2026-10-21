package com.saunak.director.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.saunak.director.ai.ModelManager
import com.saunak.director.camera.CameraUiState
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorStage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

@Composable
fun DirectorApp(vm: DirectorViewModel = viewModel()) {
    val context = LocalContext.current
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
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
    MaterialTheme(colorScheme = darkColorScheme()) {
        if (!granted) PermissionScreen { launcher.launch(Manifest.permission.CAMERA) } else CameraScreen(vm)
    }
}

@Composable
private fun PermissionScreen(request: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
        Column(Modifier.padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Director needs camera access", color = Color.White, style = MaterialTheme.typography.titleLarge)
            Text("Your captured photos and Director AI stay on-device by default.", color = Color.LightGray)
            Button(onClick = request) { Text("Allow camera") }
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
    LaunchedEffect(focusPoint) { if (focusPoint != null) { delay(1000); focusPoint = null } }

    if (ui.requestModelDownload) {
        AlertDialog(
            onDismissRequest = vm::dismissModelDownload,
            title = { Text("Enable Director AI?") },
            text = { Text("Director needs a one-time ${ModelManager.DOWNLOAD_SIZE_LABEL} on-device AI download. The camera works without it, and captured photos are not uploaded. Wi-Fi is recommended.") },
            confirmButton = { TextButton(onClick = vm::installDirectorAi) { Text("Download") } },
            dismissButton = { TextButton(onClick = vm::dismissModelDownload) { Text("Not now") } }
        )
    }

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

        Canvas(Modifier.fillMaxSize()) {
            focusPoint?.let { point -> drawCircle(Color(0xFFFFD54F), 30.dp.toPx(), point, style = Stroke(2.dp.toPx())) }
        }

        DirectorTopBar(
            status = ui.status,
            stage = ui.stage,
            progress = ui.modelProgress,
            camera = ui.camera,
            onTorch = { vm.torch(!ui.camera.torchEnabled) },
            onFlash = vm::cycleFlash,
            onExposure = { if (ui.camera.exposureSupported) showExposure = !showExposure },
            onDirector = vm::installDirectorAi,
            modifier = Modifier.align(Alignment.TopCenter)
        )

        if (showExposure && ui.camera.exposureSupported) {
            ExposureControl(ui.camera, Modifier.align(Alignment.CenterEnd).padding(end = 10.dp)) { vm.exposure(it) }
        }

        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding().padding(bottom = 18.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            ZoomRow(ui.camera, requestedZoom) { requestedZoom = it; vm.zoom(it) }
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 28.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                GalleryButton(ui.lastSavedUri) {
                    val intent = ui.lastSavedUri?.let { value ->
                        Intent(Intent.ACTION_VIEW).apply {
                            setDataAndType(Uri.parse(value), "image/jpeg")
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                    } ?: Intent(Intent.ACTION_VIEW, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                    runCatching { context.startActivity(intent) }
                        .recoverCatching { context.startActivity(Intent(Intent.ACTION_VIEW, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)) }
                }
                ShutterButton(ui.stage) {
                    haptics.performHapticFeedback(HapticFeedbackType.LongPress)
                    vm.shutterCapture()
                }
                RoundButton("↻") {
                    haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                    vm.switchCamera()
                }
            }
        }
    }
}

@Composable
private fun DirectorTopBar(
    status: String,
    stage: DirectorStage,
    progress: Float?,
    camera: CameraUiState,
    onTorch: () -> Unit,
    onFlash: () -> Unit,
    onExposure: () -> Unit,
    onDirector: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(modifier.fillMaxWidth(), color = Color.Black.copy(alpha = 0.45f)) {
        Row(
            Modifier.statusBarsPadding().padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onTorch, enabled = camera.torchSupported) {
                    Text(if (camera.torchEnabled) "⚡ON" else "⚡", color = if (camera.torchSupported) Color.White else Color.Gray)
                }
                TextButton(onClick = onFlash, enabled = camera.torchSupported) {
                    val label = when (camera.flashMode) { StillFlashMode.OFF -> "F OFF"; StillFlashMode.AUTO -> "F AUTO"; StillFlashMode.ON -> "F ON" }
                    Text(label, color = if (camera.torchSupported) Color.White else Color.Gray, style = MaterialTheme.typography.labelSmall)
                }
            }
            TextButton(onClick = onDirector, contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        val indicator = when (stage) {
                            DirectorStage.ERROR -> Color(0xFFFF6B6B)
                            DirectorStage.CAPTURING, DirectorStage.ANALYZING, DirectorStage.PLANNING, DirectorStage.APPLYING, DirectorStage.RECAPTURING, DirectorStage.CHECKING, DirectorStage.CORRECTING, DirectorStage.SAVING -> Color(0xFFFFD54F)
                            DirectorStage.READY -> Color(0xFF7DFF9B)
                            else -> Color.White
                        }
                        Box(Modifier.size(8.dp).background(indicator, CircleShape))
                        Text("Director", color = Color.White, style = MaterialTheme.typography.titleMedium)
                    }
                    Text(status, color = Color.White.copy(alpha = 0.82f), style = MaterialTheme.typography.labelSmall, maxLines = 1)
                    if (progress != null && progress in 0f..0.999f) LinearProgressIndicator(progress = { progress }, modifier = Modifier.width(140.dp).padding(top = 4.dp))
                }
            }
            TextButton(onClick = onExposure, enabled = camera.exposureSupported) {
                val ev = camera.exposureIndex * camera.exposureStepEv
                Text(if (camera.exposureSupported) String.format(java.util.Locale.US, "%+.1f", ev) else "±", color = if (camera.exposureSupported) Color.White else Color.Gray)
            }
        }
    }
}

@Composable
private fun ExposureControl(camera: CameraUiState, modifier: Modifier, onExposure: (Int) -> Unit) {
    Surface(modifier.width(56.dp).height(220.dp), color = Color.Black.copy(alpha = 0.5f), shape = CircleShape) {
        Column(Modifier.padding(vertical = 10.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceBetween) {
            TextButton(onClick = { onExposure((camera.exposureIndex + 1).coerceAtMost(camera.exposureMaxIndex)) }, enabled = camera.exposureIndex < camera.exposureMaxIndex) { Text("+", color = Color.White) }
            val ev = camera.exposureIndex * camera.exposureStepEv
            Text(String.format(java.util.Locale.US, "%+.1f", ev), color = Color.White, style = MaterialTheme.typography.labelSmall)
            TextButton(onClick = { onExposure((camera.exposureIndex - 1).coerceAtLeast(camera.exposureMinIndex)) }, enabled = camera.exposureIndex > camera.exposureMinIndex) { Text("−", color = Color.White) }
        }
    }
}

@Composable
private fun ZoomRow(camera: CameraUiState, current: Float, onZoom: (Float) -> Unit) {
    val options = remember(camera.minZoom, camera.maxZoom) {
        buildList {
            if (camera.minZoom < 0.95f) add(camera.minZoom)
            if (1f in camera.minZoom..camera.maxZoom) add(1f)
            if (2f in camera.minZoom..camera.maxZoom) add(2f)
            if (3f in camera.minZoom..camera.maxZoom) add(3f)
            if (isEmpty()) add(camera.minZoom)
        }.distinctBy { (it * 100).toInt() }
    }
    Surface(color = Color.Black.copy(alpha = 0.42f), shape = CircleShape) {
        Row(Modifier.padding(horizontal = 8.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
            options.forEach { value ->
                TextButton(onClick = { onZoom(value) }) {
                    val label = if (abs(value - value.toInt()) < 0.02f) "${value.toInt()}×" else String.format(java.util.Locale.US, "%.1f×", value)
                    Text(label, color = if (abs(current - value) < 0.12f) Color(0xFFFFD54F) else Color.White)
                }
            }
        }
    }
}

@Composable
private fun ShutterButton(stage: DirectorStage, onClick: () -> Unit) {
    val active = stage in setOf(DirectorStage.CAPTURING, DirectorStage.ANALYZING, DirectorStage.PLANNING, DirectorStage.APPLYING, DirectorStage.RECAPTURING, DirectorStage.CHECKING, DirectorStage.CORRECTING, DirectorStage.SAVING)
    Surface(
        onClick = onClick,
        shape = CircleShape,
        color = Color.White,
        border = androidx.compose.foundation.BorderStroke(if (active) 7.dp else 5.dp, if (active) Color(0xFFFFD54F) else Color.White.copy(alpha = 0.45f)),
        modifier = Modifier.size(78.dp)
    ) {}
}

@Composable
private fun GalleryButton(uri: String?, onClick: () -> Unit) {
    val context = LocalContext.current
    val thumbnail by produceState<androidx.compose.ui.graphics.ImageBitmap?>(initialValue = null, uri) {
        value = if (uri == null) null else withContext(Dispatchers.IO) {
            runCatching {
                val source = ImageDecoder.createSource(context.contentResolver, Uri.parse(uri))
                ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    val w = info.size.width; val h = info.size.height; val scale = min(1f, 180f / max(w, h).toFloat())
                    decoder.setTargetSize(max(1, (w * scale).toInt()), max(1, (h * scale).toInt()))
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }.asImageBitmap()
            }.getOrNull()
        }
    }
    Surface(onClick = onClick, shape = CircleShape, color = Color.Black.copy(alpha = 0.5f), modifier = Modifier.size(52.dp)) {
        if (thumbnail != null) Image(thumbnail!!, contentDescription = "Open last photo", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize().clip(CircleShape))
        else Box(contentAlignment = Alignment.Center) { Text("▣", color = Color.White) }
    }
}

@Composable
private fun RoundButton(label: String, onClick: () -> Unit) {
    Surface(onClick = onClick, shape = CircleShape, color = Color.Black.copy(alpha = 0.45f), modifier = Modifier.size(50.dp)) {
        Box(contentAlignment = Alignment.Center) { Text(label, color = Color.White) }
    }
}
