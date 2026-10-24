package com.saunak.director.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.util.Size
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
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
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.SierraMode
import com.saunak.director.storage.SierraCropPolicy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
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
    val ui by vm.ui.collectAsStateWithLifecycle()
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
            ui.serverConfigured && !ui.signedIn && !ui.guestMode -> SierraAccountScreen(ui, vm)
            !explainedPermission -> SierraPermissionIntro { explainedPermission = true; launcher.launch(Manifest.permission.CAMERA) }
            !granted -> SierraPermissionDenied { launcher.launch(Manifest.permission.CAMERA) }
            else -> SierraCameraScreen(vm)
        }
    }
}

@Composable
private fun SierraAccountScreen(ui: DirectorUiState, vm: DirectorViewModel) {
    var signup by remember { mutableStateOf(true) }
    var name by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var phone by remember { mutableStateOf("") }
    var dob by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    Box(Modifier.fillMaxSize().background(SierraInk)) {
        Column(
            Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().verticalScroll(rememberScrollState()).padding(horizontal = 28.dp, vertical = 34.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            SierraMark(82.dp)
            Text("SIERRA VISION", color = SierraIvory, fontSize = 24.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 4.sp)
            Text(if (signup) "Create your Sierra account" else "Welcome back", color = SierraSilver, fontSize = 15.sp)
            Spacer(Modifier.height(6.dp))

            if (signup) {
                SierraAuthField(name, { name = it }, "Name")
            }
            SierraAuthField(email, { email = it }, "Email")
            if (signup) {
                SierraAuthField(phone, { phone = it }, "Phone with country code")
                SierraAuthField(dob, { dob = it }, "Date of birth · YYYY-MM-DD")
            }
            OutlinedTextField(
                value = password,
                onValueChange = { password = it },
                label = { Text("Password") },
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp)
            )

            ui.authError?.let { Text(it, color = SierraError, fontSize = 12.sp, textAlign = TextAlign.Center) }

            Button(
                onClick = {
                    if (signup) vm.signup(name, email, phone, dob, password) else vm.login(email, password)
                },
                enabled = !ui.authBusy,
                colors = ButtonDefaults.buttonColors(containerColor = SierraGold, contentColor = Color.Black),
                shape = RoundedCornerShape(18.dp),
                modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp).height(52.dp)
            ) {
                if (ui.authBusy) CircularProgressIndicator(Modifier.size(18.dp), color = Color.Black, strokeWidth = 2.dp)
                else Text(if (signup) "Create account" else "Sign in", fontWeight = FontWeight.Bold)
            }

            TextButton(onClick = { signup = !signup }) {
                Text(if (signup) "Already have an account? Sign in" else "New to Sierra Vision? Create account", color = SierraGold)
            }

            if (signup) {
                Surface(color = SierraGraphite, shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp)) {
                    Text(
                        "To secure your account and optimize Sierra for this phone, signup registers this Sierra Vision installation plus basic app/device details such as phone model, Android/app version, locale/timezone and camera capabilities. Sierra does not collect IMEI, hardware serial, MAC address, advertising ID, contacts, calls, SMS, installed-app inventory or precise location.",
                        color = SierraSilver,
                        fontSize = 11.sp,
                        lineHeight = 17.sp,
                        modifier = Modifier.padding(15.dp)
                    )
                }
            }

            TextButton(onClick = vm::continueWithBeta) {
                Text("Continue with Sierra β · local only", color = SierraIvory)
            }
            Text("Sierra β remains available without an account or cloud connection.", color = SierraSilver, fontSize = 10.sp, textAlign = TextAlign.Center)
        }
    }
}

@Composable
private fun SierraAuthField(value: String, onValue: (String) -> Unit, label: String) {
    OutlinedTextField(
        value = value,
        onValueChange = onValue,
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp)
    )
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
    val lifecycleOwner = LocalLifecycleOwner.current
    val haptics = LocalHapticFeedback.current
    val ui by vm.ui.collectAsStateWithLifecycle()
    var focusPoint by remember { mutableStateOf<Offset?>(null) }
    var requestedZoom by remember { mutableFloatStateOf(1f) }
    var showStatus by remember { mutableStateOf(false) }

    BackHandler(ui.showGallery || ui.showTools || showStatus) {
        when {
            ui.showGallery -> vm.closeGallery()
            ui.showTools -> vm.dismissTools()
            showStatus -> showStatus = false
        }
    }

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
            onGallery = vm::openGallery,
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
        if (ui.showGallery) SierraGallery(ui, vm)
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
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            if (ui.signedIn && ui.creditsAvailable != null) {
                Surface(onClick = onSierra, color = SierraGraphite.copy(alpha = .78f), shape = RoundedCornerShape(18.dp)) {
                    Text("${ui.creditsAvailable} ◉", color = SierraSoftGold, fontSize = 11.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp))
                }
            }
            Surface(onClick = onTools, color = SierraGraphite.copy(alpha = .78f), shape = CircleShape, modifier = Modifier.size(44.dp)) {
                Box(contentAlignment = Alignment.Center) { Text("•••", color = SierraIvory, fontSize = 16.sp, letterSpacing = 1.sp) }
            }
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
        Column(Modifier.navigationBarsPadding().padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text("Camera Tools", color = SierraIvory, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                Text("✕", color = SierraIvory, modifier = Modifier.clickable { onClose() }.padding(8.dp))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                SierraTool("Flash", ui.camera.flashMode.wireName.replaceFirstChar { it.uppercase() }) { vm.cycleFlash() }
                SierraTool("Grid", if (ui.gridEnabled) "On" else "Off") { vm.toggleGrid() }
                SierraTool("Level", if (ui.levelEnabled) "On" else "Off") { vm.toggleLevel() }
            }

            HorizontalDivider(color = Color.White.copy(alpha = .08f))
            Text("Aspect ratio", color = SierraIvory, fontWeight = FontWeight.SemiBold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SierraRatioChip("Sierra", ui.aspectRatioOverride == null) { vm.setAspectRatioOverride(null) }
                SierraCropPolicy.supported.take(3).forEach { ratio -> SierraRatioChip(ratio, ui.aspectRatioOverride == ratio) { vm.setAspectRatioOverride(ratio) } }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SierraCropPolicy.supported.drop(3).forEach { ratio -> SierraRatioChip(ratio, ui.aspectRatioOverride == ratio) { vm.setAspectRatioOverride(ratio) } }
            }

            HorizontalDivider(color = Color.White.copy(alpha = .08f))
            SierraToggleRow(
                title = "Private Photography",
                subtitle = "Sierra β only · no analysis frame leaves device",
                checked = ui.privateMode,
                onChange = { vm.togglePrivateMode() }
            )
            SierraToggleRow(
                title = "Keep Original",
                subtitle = "Also save the untouched full-quality CameraX JPEG",
                checked = ui.keepOriginal,
                onChange = { vm.toggleKeepOriginal() }
            )
            Text("Sierra finals save to Pictures/Sierra Vision. Originals, when enabled, save separately under Originals.", color = SierraSilver, fontSize = 11.sp, lineHeight = 16.sp)
        }
    }
}

@Composable
private fun SierraRatioChip(label: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(label, fontSize = 10.sp) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = SierraGold,
            selectedLabelColor = Color.Black,
            containerColor = Color.White.copy(alpha = .04f),
            labelColor = SierraIvory
        )
    )
}

@Composable
private fun SierraToggleRow(title: String, subtitle: String, checked: Boolean, onChange: () -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, color = SierraIvory, fontWeight = FontWeight.SemiBold)
            Text(subtitle, color = SierraSilver, fontSize = 11.sp, lineHeight = 15.sp)
        }
        Switch(checked = checked, onCheckedChange = { onChange() }, colors = SwitchDefaults.colors(checkedThumbColor = Color.Black, checkedTrackColor = SierraGold))
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
                if (ui.signedIn) {
                    SierraStatusRow("Account", ui.accountName ?: "Signed in")
                    SierraStatusRow("AI credits", ui.creditsAvailable?.toString() ?: "Syncing")
                } else {
                    SierraStatusRow("Account", if (ui.guestMode) "Sierra β guest" else "Not signed in")
                }
                SierraStatusRow("Sierra AI", if (ui.aiAvailable && !ui.privateMode) "Available" else if (ui.privateMode) "Disabled by Private Photography" else "Unavailable")
                SierraStatusRow("Sierra β", "Active")
                SierraStatusRow("Current intelligence", if (ui.sierraMode == SierraMode.AI) "Sierra AI + Sierra β" else "Sierra β")
                SierraStatusRow("Composition", ui.aspectRatioOverride?.let { "$it locked" } ?: "Sierra Auto")
                SierraStatusRow("Saving", if (ui.keepOriginal) "Final + Original" else "Sierra Final")
                HorizontalDivider(color = Color.White.copy(alpha = .08f))
                Text("Sierra AI directs the photograph when available. Sierra β always validates and executes camera controls locally, and takes over planning automatically when cloud intelligence is unavailable or too slow.", color = SierraSilver, fontSize = 12.sp, lineHeight = 18.sp)
                if (ui.privateMode) Text("Private Photography is active. No cloud analysis will be attempted.", color = SierraSoftGold, fontSize = 12.sp)
                TextButton(onClick = { vm.togglePrivateMode() }) { Text(if (ui.privateMode) "Disable Private Photography" else "Use Sierra β privately", color = SierraGold) }
                if (ui.signedIn) TextButton(onClick = { onClose(); vm.signOut() }) { Text("Sign out", color = SierraSilver) }
            }
        }
    }
}

@Composable
private fun SierraGallery(ui: DirectorUiState, vm: DirectorViewModel) {
    val context = LocalContext.current
    var selected by remember { mutableStateOf<String?>(null) }

    Box(Modifier.fillMaxSize().background(SierraInk)) {
        Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Column {
                    Text("SIERRA VISION", color = SierraIvory, letterSpacing = 3.sp, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Text("Gallery", color = SierraSilver, fontSize = 12.sp)
                }
                Text("✕", color = SierraIvory, modifier = Modifier.clickable { vm.closeGallery() }.padding(12.dp), fontSize = 20.sp)
            }
            if (ui.galleryItems.isEmpty()) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        SierraMark(62.dp)
                        Text("No Sierra photos yet", color = SierraIvory, fontWeight = FontWeight.SemiBold)
                        Text("Your finished photographs will appear here.", color = SierraSilver, fontSize = 12.sp)
                    }
                }
            } else {
                LazyVerticalGrid(
                    columns = GridCells.Fixed(3),
                    modifier = Modifier.fillMaxSize().padding(horizontal = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(3.dp),
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    items(ui.galleryItems, key = { it }) { value -> SierraGalleryThumbnail(value) { selected = value } }
                }
            }
        }

        selected?.let { value ->
            SierraPhotoViewer(
                uriString = value,
                onClose = { selected = null },
                onOpen = {
                    runCatching {
                        context.startActivity(Intent(Intent.ACTION_VIEW).apply {
                            setDataAndType(Uri.parse(value), "image/jpeg")
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        })
                    }
                },
                onShare = {
                    runCatching {
                        context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                            type = "image/jpeg"
                            putExtra(Intent.EXTRA_STREAM, Uri.parse(value))
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }, "Share Sierra photo"))
                    }
                }
            )
        }
    }
}

@Composable
private fun SierraGalleryThumbnail(uriString: String, onClick: () -> Unit) {
    val bitmap = rememberMediaThumbnail(uriString, 480)
    Surface(onClick = onClick, color = Color(0xFF17191F), shape = RoundedCornerShape(3.dp), modifier = Modifier.fillMaxWidth().height(150.dp)) {
        if (bitmap != null) Image(bitmap.asImageBitmap(), contentDescription = "Sierra photo", modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 1.5.dp, color = SierraGold) }
    }
}

@Composable
private fun SierraPhotoViewer(uriString: String, onClose: () -> Unit, onOpen: () -> Unit, onShare: () -> Unit) {
    val bitmap = rememberMediaThumbnail(uriString, 1600)
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (bitmap != null) {
            Image(bitmap.asImageBitmap(), contentDescription = "Sierra photograph", modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        } else {
            CircularProgressIndicator(Modifier.align(Alignment.Center), color = SierraGold)
        }
        Row(Modifier.fillMaxWidth().statusBarsPadding().padding(16.dp), horizontalArrangement = Arrangement.SpaceBetween) {
            SierraRoundAction("‹", onClose)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Surface(onClick = onShare, color = SierraGraphite.copy(alpha = .86f), shape = RoundedCornerShape(18.dp)) { Text("Share", Modifier.padding(horizontal = 14.dp, vertical = 10.dp), color = SierraIvory) }
                Surface(onClick = onOpen, color = SierraGraphite.copy(alpha = .86f), shape = RoundedCornerShape(18.dp)) { Text("Open", Modifier.padding(horizontal = 14.dp, vertical = 10.dp), color = SierraIvory) }
            }
        }
    }
}

@Composable
private fun rememberMediaThumbnail(uriString: String, px: Int): Bitmap? {
    val context = LocalContext.current
    val bitmap by produceState<Bitmap?>(initialValue = null, uriString, px) {
        value = withContext(Dispatchers.IO) {
            runCatching { context.contentResolver.loadThumbnail(Uri.parse(uriString), Size(px, px), null) }.getOrNull()
        }
    }
    return bitmap
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
    Surface(onClick = onClick, color = Color.White.copy(alpha = .05f), shape = RoundedCornerShape(16.dp), modifier = Modifier.width(100.dp)) {
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
    "4:3" -> 4f to 3f
    "3:2" -> 3f to 2f
    "4:5" -> 4f to 5f
    "1:1" -> 1f to 1f
    "16:9" -> 16f to 9f
    "9:16" -> 9f to 16f
    "2.39:1" -> 2.39f to 1f
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
