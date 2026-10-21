package com.saunak.director.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.saunak.director.ai.ModelManager
import com.saunak.director.director.DirectorStage
import kotlinx.coroutines.delay

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
            Text("Photos and AI analysis stay on this device by default.", color = Color.LightGray)
            Button(onClick = request) { Text("Allow camera") }
        }
    }
}

@Composable
private fun CameraScreen(vm: DirectorViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val ui by vm.ui.collectAsStateWithLifecycle()
    var zoom by remember { mutableFloatStateOf(1f) }
    var torch by remember { mutableStateOf(false) }
    var focusPoint by remember { mutableStateOf<Offset?>(null) }
    var showExposure by remember { mutableStateOf(false) }
    var exposureIndex by remember { mutableFloatStateOf(0f) }

    LaunchedEffect(focusPoint) { if (focusPoint != null) { delay(1200); focusPoint = null } }

    if (ui.requestModelDownload) {
        AlertDialog(
            onDismissRequest = vm::dismissModelDownload,
            title = { Text("Enable Director AI?") },
            text = { Text("Director needs a one-time ${ModelManager.DOWNLOAD_SIZE_LABEL} on-device AI download. The normal camera already works, and your photos are not uploaded. Wi-Fi is recommended.") },
            confirmButton = { TextButton(onClick = vm::installDirectorAi) { Text("Download") } },
            dismissButton = { TextButton(onClick = vm::dismissModelDownload) { Text("Not now") } }
        )
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            factory = { ctx -> PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER; implementationMode = PreviewView.ImplementationMode.PERFORMANCE; vm.bind(lifecycleOwner, this) } },
            modifier = Modifier.fillMaxSize()
                .pointerInput(Unit) { detectTapGestures { point -> focusPoint = point; vm.focus(point.x, point.y) } }
                .pointerInput(zoom) { detectTransformGestures { _, _, gestureZoom, _ -> if (gestureZoom != 1f) { zoom = (zoom * gestureZoom).coerceIn(1f, 10f); vm.zoom(zoom) } } }
        )

        Canvas(Modifier.fillMaxSize()) {
            focusPoint?.let { point -> drawCircle(Color(0xFFFFD54F), 30.dp.toPx(), point, style = Stroke(2.dp.toPx())) }
        }

        DirectorTopBar(ui.status, ui.stage, ui.modelProgress, torch,
            onTorch = { torch = !torch; vm.torch(torch) },
            onExposure = { showExposure = !showExposure },
            onDirector = vm::installDirectorAi,
            modifier = Modifier.align(Alignment.TopCenter))

        if (showExposure) {
            Surface(Modifier.align(Alignment.CenterEnd).padding(end = 10.dp).width(48.dp).height(220.dp), color = Color.Black.copy(alpha = 0.45f), shape = CircleShape) {
                Column(Modifier.padding(vertical = 12.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceBetween) {
                    TextButton(onClick = { exposureIndex = (exposureIndex + 1f).coerceAtMost(6f); vm.exposure(exposureIndex.toInt()) }) { Text("+", color = Color.White) }
                    Text(exposureIndex.toInt().toString(), color = Color.White, style = MaterialTheme.typography.labelSmall)
                    TextButton(onClick = { exposureIndex = (exposureIndex - 1f).coerceAtLeast(-6f); vm.exposure(exposureIndex.toInt()) }) { Text("−", color = Color.White) }
                }
            }
        }

        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding().padding(bottom = 18.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
            ZoomRow(zoom) { zoom = it; vm.zoom(it) }
            Row(Modifier.fillMaxWidth().padding(horizontal = 28.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                RoundButton("▣") { runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)) } }
                ShutterButton { vm.manualCapture() }
                RoundButton("↻") { vm.switchCamera() }
            }
        }
    }
}

@Composable
private fun DirectorTopBar(status: String, stage: DirectorStage, progress: Float?, torch: Boolean, onTorch: () -> Unit, onExposure: () -> Unit, onDirector: () -> Unit, modifier: Modifier = Modifier) {
    Surface(modifier.fillMaxWidth(), color = Color.Black.copy(alpha = 0.45f)) {
        Row(Modifier.statusBarsPadding().padding(horizontal = 12.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            TextButton(onClick = onTorch) { Text(if (torch) "⚡ ON" else "⚡", color = Color.White) }
            TextButton(onClick = onDirector, contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        val indicator = when (stage) { DirectorStage.ERROR -> Color(0xFFFF6B6B); DirectorStage.PLANNING, DirectorStage.APPLYING, DirectorStage.CHECKING, DirectorStage.CORRECTING -> Color(0xFFFFD54F); else -> Color.White }
                        Box(Modifier.size(8.dp).background(indicator, CircleShape)); Text("Director", color = Color.White, style = MaterialTheme.typography.titleMedium)
                    }
                    Text(status, color = Color.White.copy(alpha = 0.82f), style = MaterialTheme.typography.labelSmall)
                    if (progress != null && progress in 0f..0.999f) LinearProgressIndicator(progress = { progress }, modifier = Modifier.width(140.dp).padding(top = 4.dp))
                }
            }
            TextButton(onClick = onExposure) { Text("±", color = Color.White) }
        }
    }
}

@Composable private fun ZoomRow(current: Float, onZoom: (Float) -> Unit) {
    Surface(color = Color.Black.copy(alpha = 0.42f), shape = CircleShape) {
        Row(Modifier.padding(horizontal = 8.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            listOf(1f, 2f, 3f).forEach { value -> TextButton(onClick = { onZoom(value) }) { Text("${value.toInt()}×", color = if (kotlin.math.abs(current - value) < 0.15f) Color(0xFFFFD54F) else Color.White) } }
        }
    }
}

@Composable private fun ShutterButton(onClick: () -> Unit) { Surface(onClick = onClick, shape = CircleShape, color = Color.White, border = androidx.compose.foundation.BorderStroke(5.dp, Color.White.copy(alpha = 0.45f)), modifier = Modifier.size(78.dp)) {} }
@Composable private fun RoundButton(label: String, onClick: () -> Unit) { Surface(onClick = onClick, shape = CircleShape, color = Color.Black.copy(alpha = 0.45f), modifier = Modifier.size(50.dp)) { Box(contentAlignment = Alignment.Center) { Text(label, color = Color.White) } } }
