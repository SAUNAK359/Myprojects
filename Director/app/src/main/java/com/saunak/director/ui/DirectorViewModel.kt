package com.saunak.director.ui

import android.app.Application
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.saunak.director.ai.DeviceConfigSkill
import com.saunak.director.ai.SierraServerVlmRuntime
import com.saunak.director.backend.DeviceProfileCollector
import com.saunak.director.backend.SierraServerClient
import com.saunak.director.camera.CameraEngine
import com.saunak.director.camera.StillFlashMode
import com.saunak.director.director.DirectorOrchestrator
import com.saunak.director.director.DirectorStage
import com.saunak.director.director.DirectorUiState
import com.saunak.director.director.SierraMode
import com.saunak.director.platform.ThermalMonitor
import com.saunak.director.skills.SkillEngine
import com.saunak.director.storage.PhotoRepository
import com.saunak.director.storage.RecipeMemory
import com.saunak.director.storage.SierraCropPolicy
import com.saunak.director.tools.ToolExecutor
import java.util.Locale
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class DirectorViewModel(app: Application) : AndroidViewModel(app) {
    private val camera = CameraEngine(app)
    private val server = SierraServerClient(app)
    private val deviceProfile = DeviceProfileCollector(app)
    private val recipeMemory = RecipeMemory(app)
    private val deviceConfigSkill = DeviceConfigSkill(app)
    private val runtime = SierraServerVlmRuntime(app, server = server, deviceSkill = deviceConfigSkill)
    private val repository = PhotoRepository(app)
    private val toolExecutor = ToolExecutor(camera)
    private val preferences = app.getSharedPreferences("sierra_vision_preferences", Application.MODE_PRIVATE)
    private val initialPrivateMode = preferences.getBoolean("private_mode", false)
    private val initialGuestMode = preferences.getBoolean("guest_mode", false)
    private val initialAspectRatioOverride = preferences.getString("aspect_ratio_override", null)?.takeIf(SierraCropPolicy::isSupported)
    private val initialKeepOriginal = preferences.getBoolean("keep_original", false)
    private val initialSession = server.session()
    private val directAiReadyAtStart = runtime.isReady() && initialSession == null

    private val orchestrator = DirectorOrchestrator(
        camera = camera,
        vlm = runtime,
        skills = SkillEngine(app),
        tools = toolExecutor,
        storage = repository,
        recipeMemory = recipeMemory,
        thermal = ThermalMonitor(app)
    ) { state ->
        val current = _ui.value
        _ui.value = state.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(state.camera),
            recipeCount = recipeMemory.count(),
            showSettings = current.showSettings,
            showTools = current.showTools,
            showGallery = current.showGallery,
            galleryItems = current.galleryItems,
            gridEnabled = current.gridEnabled,
            levelEnabled = current.levelEnabled,
            privateMode = current.privateMode,
            aspectRatioOverride = current.aspectRatioOverride,
            keepOriginal = current.keepOriginal,
            serverConfigured = current.serverConfigured,
            signedIn = current.signedIn,
            guestMode = current.guestMode,
            accountName = current.accountName,
            creditsAvailable = server.session()?.creditsAvailable ?: current.creditsAvailable,
            authBusy = current.authBusy,
            authError = current.authError
        )
        if (state.lastSavedUri != null && state.stage == DirectorStage.READY) refreshGallery()
    }

    private val _ui = MutableStateFlow(
        DirectorUiState(
            aiAvailable = runtime.isReady() && !initialPrivateMode,
            recipeCount = recipeMemory.count(),
            status = if (runtime.isReady() && !initialPrivateMode) "Sierra AI + β ready" else "Sierra β ready",
            sierraMode = if (runtime.isReady() && !initialPrivateMode) SierraMode.AI else SierraMode.BETA,
            privateMode = initialPrivateMode,
            aspectRatioOverride = initialAspectRatioOverride,
            aspectRatio = initialAspectRatioOverride ?: "4:3",
            keepOriginal = initialKeepOriginal,
            serverConfigured = server.configured,
            signedIn = initialSession != null,
            guestMode = (initialGuestMode || directAiReadyAtStart) && initialSession == null,
            accountName = initialSession?.displayName,
            creditsAvailable = initialSession?.creditsAvailable
        )
    )
    val ui: StateFlow<DirectorUiState> = _ui.asStateFlow()

    private var previewView: PreviewView? = null
    private var owner: LifecycleOwner? = null
    @Volatile private var cameraBound = false

    init {
        orchestrator.setCloudEnabled(runtime.isReady() && !initialPrivateMode)
        orchestrator.setAspectRatioOverride(initialAspectRatioOverride)
        orchestrator.setKeepOriginal(initialKeepOriginal)
        refreshGallery()
        refreshCredits()
        retryPendingSettlement()
        viewModelScope.launch {
            val recovered = orchestrator.recoverPendingReferences()
            if (recovered.isNotEmpty()) {
                _ui.value = _ui.value.copy(
                    status = "Recovered ${recovered.size} protected photo${if (recovered.size == 1) "" else "s"}",
                    lastSavedUri = recovered.last().toString()
                )
                refreshGallery()
            }
        }
    }

    fun bind(owner: LifecycleOwner, view: PreviewView) {
        if (cameraBound && previewView === view && this.owner === owner) return
        previewView = view
        this.owner = owner
        viewModelScope.launch {
            runCatching { camera.bind(owner, view) }
                .onSuccess {
                    cameraBound = true
                    withContext(Dispatchers.IO) {
                        runCatching { deviceConfigSkill.refresh(camera.capabilities(), camera.snapshot()) }
                    }
                    refreshCamera()
                    syncDeviceProfile()
                    launch { ensureDeviceFoundation() }
                }
                .onFailure {
                    cameraBound = false
                    _ui.value = _ui.value.copy(stage = DirectorStage.ERROR, status = "Camera unavailable", error = it.message)
                }
        }
    }

    fun signup(name: String, email: String, phone: String, dobIso: String, password: String) {
        if (_ui.value.authBusy) return
        _ui.value = _ui.value.copy(authBusy = true, authError = null)
        viewModelScope.launch {
            server.signup(name.trim(), email.trim(), phone.trim(), dobIso.trim(), password, deviceProfile.collect())
                .onSuccess { session ->
                    preferences.edit().putBoolean("guest_mode", false).apply()
                    orchestrator.setCloudEnabled(!_ui.value.privateMode)
                    _ui.value = _ui.value.copy(
                        signedIn = true,
                        guestMode = false,
                        accountName = session.displayName,
                        creditsAvailable = session.creditsAvailable,
                        aiAvailable = !_ui.value.privateMode,
                        sierraMode = if (_ui.value.privateMode) SierraMode.BETA else SierraMode.AI,
                        authBusy = false,
                        authError = null,
                        status = if (_ui.value.privateMode) "Sierra β Private • on-device only" else "Sierra AI + β ready"
                    )
                    syncDeviceProfile()
                    launch { ensureDeviceFoundation() }
                }
                .onFailure { error ->
                    _ui.value = _ui.value.copy(authBusy = false, authError = error.message ?: "Could not create account")
                }
        }
    }

    fun login(email: String, password: String) {
        if (_ui.value.authBusy) return
        _ui.value = _ui.value.copy(authBusy = true, authError = null)
        viewModelScope.launch {
            server.login(email.trim(), password)
                .onSuccess { session ->
                    preferences.edit().putBoolean("guest_mode", false).apply()
                    orchestrator.setCloudEnabled(!_ui.value.privateMode)
                    _ui.value = _ui.value.copy(
                        signedIn = true,
                        guestMode = false,
                        accountName = session.displayName,
                        creditsAvailable = session.creditsAvailable,
                        aiAvailable = !_ui.value.privateMode,
                        sierraMode = if (_ui.value.privateMode) SierraMode.BETA else SierraMode.AI,
                        authBusy = false,
                        authError = null,
                        status = if (_ui.value.privateMode) "Sierra β Private • on-device only" else "Sierra AI + β ready"
                    )
                    syncDeviceProfile()
                    launch { ensureDeviceFoundation() }
                }
                .onFailure { error ->
                    _ui.value = _ui.value.copy(authBusy = false, authError = error.message ?: "Could not sign in")
                }
        }
    }

    fun continueWithBeta() {
        preferences.edit().putBoolean("guest_mode", true).apply()
        orchestrator.setCloudEnabled(false)
        _ui.value = _ui.value.copy(
            guestMode = true,
            signedIn = false,
            aiAvailable = false,
            sierraMode = SierraMode.BETA,
            authBusy = false,
            authError = null,
            status = "Sierra β ready • local only"
        )
    }

    fun signOut() {
        orchestrator.cancelForUserAction()
        server.logoutLocal()
        val directReady = runtime.isReady() && !_ui.value.privateMode
        preferences.edit().putBoolean("guest_mode", directReady).apply()
        orchestrator.setCloudEnabled(directReady)
        _ui.value = _ui.value.copy(
            signedIn = false,
            guestMode = directReady,
            accountName = null,
            creditsAvailable = null,
            aiAvailable = directReady,
            sierraMode = if (directReady) SierraMode.AI else SierraMode.BETA,
            status = if (directReady) "Sierra AI + β ready" else "Signed out • Sierra β available"
        )
        if (directReady) viewModelScope.launch { ensureDeviceFoundation() }
    }

    fun primaryCapture() {
        viewModelScope.launch {
            val preparedBeforePress = orchestrator.isPrepared()
            val aiWanted = runtime.isReady() && !_ui.value.privateMode
            val foundation = if (aiWanted && !preparedBeforePress) ensureDeviceFoundation() else Result.success(Unit)
            val useLocalForThisShot = aiWanted && foundation.isFailure
            if (useLocalForThisShot) {
                orchestrator.setCloudEnabled(false)
                _ui.value = _ui.value.copy(
                    status = "Sierra device foundation unavailable • β taking over this shot",
                    aiFallbackReason = foundation.exceptionOrNull()?.message?.take(120)
                )
            }
            try {
                val saved = orchestrator.onPrimaryPressed()
                if (preparedBeforePress) {
                    if (saved != null) {
                        val captureId = "capture-${UUID.randomUUID()}"
                        runtime.completeFinalCapture(captureId)
                            .onSuccess { refreshCredits() }
                            .onFailure {
                                _ui.value = _ui.value.copy(status = "Sierra photo saved • account sync pending")
                            }
                    } else {
                        runtime.cancelCloudJob()
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (oom: OutOfMemoryError) {
                runtime.cancel()
                _ui.value = _ui.value.copy(
                    stage = DirectorStage.ERROR,
                    aiAvailable = runtime.isReady() && !_ui.value.privateMode,
                    status = "Sierra memory pressure • try again",
                    error = "Analysis used too much memory"
                )
            } catch (t: Throwable) {
                runtime.cancel()
                _ui.value = _ui.value.copy(
                    stage = DirectorStage.ERROR,
                    aiAvailable = runtime.isReady() && !_ui.value.privateMode,
                    status = "Sierra could not prepare the shot",
                    error = t.message ?: t::class.java.simpleName
                )
            } finally {
                if (useLocalForThisShot) orchestrator.setCloudEnabled(runtime.isReady() && !_ui.value.privateMode)
            }
        }
    }

    fun focus(x: Float, y: Float) {
        orchestrator.cancelForUserAction()
        previewView?.let { camera.tapToFocus(it.meteringPointFactory, x, y) }
        refreshCameraSoon("Focus adjusted")
    }

    fun zoom(value: Float) {
        orchestrator.cancelForUserAction()
        camera.userSetZoom(value)
        refreshCameraSoon("Zoom ${String.format(Locale.US, "%.1f", value)}×")
    }

    fun exposure(index: Int) {
        orchestrator.cancelForUserAction()
        camera.userSetExposureIndex(index)
        refreshCameraSoon("Exposure adjusted")
    }

    fun torch(enabled: Boolean) {
        orchestrator.cancelForUserAction()
        camera.userSetTorch(enabled)
        refreshCameraSoon(if (enabled) "Torch on" else "Torch off")
    }

    fun cycleFlash() {
        orchestrator.cancelForUserAction()
        val next = when (_ui.value.camera.flashMode) {
            StillFlashMode.OFF -> StillFlashMode.AUTO
            StillFlashMode.AUTO -> StillFlashMode.ON
            StillFlashMode.ON -> StillFlashMode.OFF
        }
        camera.userSetFlash(next)
        refreshCameraSoon("Flash ${next.wireName}")
    }

    fun switchCamera() {
        orchestrator.cancelForUserAction()
        runCatching { camera.switchLens() }
            .onSuccess { switched ->
                if (switched) {
                    viewModelScope.launch(Dispatchers.IO) {
                        runCatching { deviceConfigSkill.refresh(camera.capabilities(), camera.snapshot()) }
                    }
                    syncDeviceProfile()
                    viewModelScope.launch { ensureDeviceFoundation() }
                }
                refreshCamera(if (switched) "Camera switched" else "Other camera unavailable")
            }
            .onFailure { _ui.value = _ui.value.copy(error = it.message, status = "Camera switch failed") }
    }

    fun togglePrivateMode() {
        val enabled = !_ui.value.privateMode
        orchestrator.cancelForUserAction()
        orchestrator.setCloudEnabled(!enabled && runtime.isReady())
        preferences.edit().putBoolean("private_mode", enabled).apply()
        val aiReady = !enabled && runtime.isReady()
        _ui.value = _ui.value.copy(
            privateMode = enabled,
            aiAvailable = aiReady,
            sierraMode = if (aiReady) SierraMode.AI else SierraMode.BETA,
            stage = DirectorStage.READY,
            status = if (enabled) "Sierra β Private • on-device only" else if (aiReady) "Sierra AI + β ready" else "Sierra β ready",
            error = null
        )
        if (aiReady) viewModelScope.launch { ensureDeviceFoundation() }
    }

    fun setAspectRatioOverride(value: String?) {
        val safe = value?.takeIf(SierraCropPolicy::isSupported)
        orchestrator.cancelForUserAction()
        orchestrator.setAspectRatioOverride(safe)
        preferences.edit().apply {
            if (safe == null) remove("aspect_ratio_override") else putString("aspect_ratio_override", safe)
        }.apply()
        _ui.value = _ui.value.copy(
            aspectRatioOverride = safe,
            aspectRatio = safe ?: _ui.value.aspectRatio,
            stage = DirectorStage.READY,
            status = if (safe == null) "Aspect ratio • Sierra Auto" else "Aspect ratio • $safe",
            error = null
        )
    }

    fun toggleKeepOriginal() {
        val enabled = !_ui.value.keepOriginal
        orchestrator.setKeepOriginal(enabled)
        preferences.edit().putBoolean("keep_original", enabled).apply()
        _ui.value = _ui.value.copy(
            keepOriginal = enabled,
            status = if (enabled) "Originals will also be saved" else "Saving Sierra final only"
        )
    }

    fun openGallery() {
        refreshGallery()
        _ui.value = _ui.value.copy(showGallery = true, showTools = false, showSettings = false)
    }
    fun closeGallery() { _ui.value = _ui.value.copy(showGallery = false) }

    fun refreshGallery() {
        viewModelScope.launch(Dispatchers.IO) {
            val items = runCatching { repository.recentFinals() }.getOrDefault(emptyList()).map { it.toString() }
            withContext(Dispatchers.Main) { _ui.value = _ui.value.copy(galleryItems = items) }
        }
    }

    fun toggleSettings() { _ui.value = _ui.value.copy(showSettings = !_ui.value.showSettings, showTools = false) }
    fun dismissSettings() { _ui.value = _ui.value.copy(showSettings = false) }
    fun toggleTools() { _ui.value = _ui.value.copy(showTools = !_ui.value.showTools, showSettings = false) }
    fun dismissTools() { _ui.value = _ui.value.copy(showTools = false) }
    fun toggleGrid() { _ui.value = _ui.value.copy(gridEnabled = !_ui.value.gridEnabled) }
    fun toggleLevel() { _ui.value = _ui.value.copy(levelEnabled = !_ui.value.levelEnabled) }

    private fun refreshCamera(status: String? = null) {
        val current = _ui.value
        val aiReady = runtime.isReady() && !current.privateMode
        _ui.value = current.copy(
            camera = runCatching { camera.uiState() }.getOrDefault(current.camera),
            status = status ?: if (current.privateMode) "Sierra β Private • on-device only" else if (aiReady) "Sierra AI + β ready" else "Sierra β ready",
            aiAvailable = aiReady,
            recipeCount = recipeMemory.count(),
            sierraMode = if (aiReady) SierraMode.AI else SierraMode.BETA,
            creditsAvailable = server.session()?.creditsAvailable ?: current.creditsAvailable,
            stage = DirectorStage.READY
        )
    }

    private fun refreshCredits() {
        if (!server.authenticated) return
        viewModelScope.launch {
            server.credits().onSuccess { value -> _ui.value = _ui.value.copy(creditsAvailable = value.available) }
        }
    }

    private fun retryPendingSettlement() {
        if (!server.authenticated) return
        viewModelScope.launch {
            runtime.retryPendingSettlement().onSuccess { refreshCredits() }
        }
    }

    private suspend fun ensureDeviceFoundation(): Result<Unit> {
        if (!cameraBound || _ui.value.privateMode || !runtime.isReady()) return Result.success(Unit)
        return withContext(Dispatchers.IO) {
            runCatching {
                val caps = camera.capabilities()
                val profile = deviceConfigSkill.foundationProfile(caps)
                runtime.syncDeviceFoundation(profile).getOrThrow()
            }
        }
    }

    private fun syncDeviceProfile() {
        if (!server.authenticated) return
        viewModelScope.launch {
            val cameraProfile = if (cameraBound) runCatching {
                val caps = camera.capabilities()
                mapOf(
                    "lens_facing" to caps.lensFacing.toString(),
                    "zoom_min" to caps.minZoom.toString(),
                    "zoom_max" to caps.maxZoom.toString(),
                    "exposure_ev_min" to caps.minExposureEv.toString(),
                    "exposure_ev_max" to caps.maxExposureEv.toString(),
                    "flash" to caps.hasFlash.toString(),
                    "torch" to caps.torchSupported.toString(),
                    "focus_metering" to caps.focusMeteringSupported.toString(),
                    "manual_sensor" to caps.manualSensorSupported.toString(),
                    "raw" to caps.rawSupported.toString(),
                    "focal_lengths_mm" to caps.focalLengthsMm.joinToString(","),
                    "extensions" to caps.extensions.joinToString(",") { it.wireName }
                )
            }.getOrNull() else null
            server.syncDevice(deviceProfile.collect(cameraProfile))
        }
    }

    private fun refreshCameraSoon(status: String) {
        viewModelScope.launch { delay(80); refreshCamera(status) }
    }

    fun onAppBackgrounded() {
        orchestrator.cancelForUserAction()
        runtime.cancel()
        viewModelScope.launch {
            runCatching { camera.releaseToSystem() }
            cameraBound = false
        }
    }

    fun onAppForegrounded() {
        val currentOwner = owner
        val currentView = previewView
        if (!cameraBound && currentOwner != null && currentView != null) {
            bind(currentOwner, currentView)
        } else {
            refreshCamera()
            viewModelScope.launch { ensureDeviceFoundation() }
        }
        refreshGallery()
        refreshCredits()
        retryPendingSettlement()
        syncDeviceProfile()
    }

    override fun onCleared() {
        orchestrator.cancelForUserAction()
        runtime.close()
        camera.close()
        super.onCleared()
    }
}
