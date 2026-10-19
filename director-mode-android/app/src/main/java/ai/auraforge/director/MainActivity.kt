package ai.auraforge.director

import ai.auraforge.director.camera.CameraCapabilities
import ai.auraforge.director.camera.CameraEngine
import ai.auraforge.director.capture.BurstCaptureController
import ai.auraforge.director.core.DirectorFrameState
import ai.auraforge.director.core.ExtensionPreference
import ai.auraforge.director.core.PhotoPlan
import ai.auraforge.director.core.PlanReadiness
import ai.auraforge.director.core.SceneState
import ai.auraforge.director.core.ThermalGovernor
import ai.auraforge.director.databinding.ActivityMainBinding
import ai.auraforge.director.director.AutonomousPhotoDirector
import ai.auraforge.director.director.PhotoPlanValidator
import ai.auraforge.director.director.PhotoReadinessEvaluator
import ai.auraforge.director.ml.FrameAnalyzer
import ai.auraforge.director.ml.VisionPipeline
import ai.auraforge.director.quality.ImageQualityAnalyzer
import ai.auraforge.director.sensors.SensorFusion
import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var captureExecutor: ExecutorService
    private lateinit var vlmExecutor: ExecutorService

    private lateinit var sensorFusion: SensorFusion
    private lateinit var vision: VisionPipeline
    private lateinit var imageQuality: ImageQualityAnalyzer
    private lateinit var frameAnalyzer: FrameAnalyzer
    private lateinit var cameraEngine: CameraEngine
    private lateinit var captureController: BurstCaptureController
    private lateinit var thermalGovernor: ThermalGovernor
    private lateinit var photoDirector: AutonomousPhotoDirector

    private val planValidator = PhotoPlanValidator()
    private val readinessEvaluator = PhotoReadinessEvaluator()

    private var directorEnabled = true
    private var gridEnabled = false
    private var cameraReady = false
    private var native4KAvailable = false
    private var lastSceneSignature = ""
    private var lastVlmSignature = ""
    private var lastVlmRequestAt = 0L
    private var lastAppliedPlanFingerprint = ""
    private var latestReadiness = PlanReadiness(false, 0f, "AI starting")
    private var externalOutputUri: Uri? = null
    private var extensionNames: Set<String> = emptySet()

    @Volatile private var activePlan: PhotoPlan? = null
    private val vlmInFlight = AtomicBoolean(false)

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        if (result[Manifest.permission.CAMERA] == true) startSystem()
        else Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.BLACK
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        externalOutputUri = if (intent.action == MediaStore.ACTION_IMAGE_CAPTURE) {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(MediaStore.EXTRA_OUTPUT) as? Uri
        } else null

        sensorFusion = SensorFusion(this)
        imageQuality = ImageQualityAnalyzer()
        analysisExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "Director-Analysis").apply { priority = Thread.NORM_PRIORITY + 1 } }
        captureExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "Director-Capture") }
        vlmExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "Director-VLM").apply { priority = Thread.NORM_PRIORITY - 1 } }

        vision = VisionPipeline(this, preferGpu = true)
        photoDirector = AutonomousPhotoDirector(this)
        frameAnalyzer = FrameAnalyzer(vision, imageQuality, sensorFusion, ::onScene)
        cameraEngine = CameraEngine(this, this, binding.previewView, frameAnalyzer, analysisExecutor)
        captureController = BurstCaptureController(this, { cameraEngine.imageCapture }, captureExecutor, imageQuality)

        thermalGovernor = ThermalGovernor(this) { level, label, interval ->
            frameAnalyzer.targetIntervalMs = interval
            runOnUiThread {
                binding.thermalBadge.text = "● $label"
                binding.thermalBadge.setTextColor(
                    getColorCompat(
                        when {
                            level >= 3 -> R.color.director_danger
                            level >= 1 -> R.color.director_amber
                            else -> R.color.director_green
                        }
                    )
                )
            }
        }

        setupUi()
        requestPermissionsOrStart()
    }

    private fun requestPermissionsOrStart() {
        val requested = buildList {
            add(Manifest.permission.CAMERA)
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }.filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (requested.isEmpty()) startSystem() else permissionLauncher.launch(requested.toTypedArray())
    }

    private fun startSystem() {
        sensorFusion.start()
        thermalGovernor.start()
        analysisExecutor.execute { vision.initialize() }
        vlmExecutor.execute { photoDirector.initialize() }
        cameraEngine.start(
            onReady = ::onCameraReady,
            onError = { t -> runOnUiThread { Toast.makeText(this, "Camera failed: ${t.message}", Toast.LENGTH_LONG).show() } }
        )
    }

    private fun onCameraReady(caps: CameraCapabilities) {
        cameraReady = true
        native4KAvailable = caps.native4K
        activePlan = null
        lastAppliedPlanFingerprint = ""
        val size = caps.maxJpegSize
        runOnUiThread {
            binding.resolutionBadge.text = if (caps.native4K) "NATIVE 4K+" else size?.let { "MAX ${it.width}×${it.height}" } ?: "MAX SENSOR"
            binding.resolutionBadge.setTextColor(getColorCompat(if (caps.native4K) R.color.director_cyan else R.color.director_amber))
            configureLensButtons()
            setShutterReady(false)
        }
    }

    private fun setupUi() {
        binding.previewView.setOnTouchListener(CameraTouchHandler())
        binding.torchButton.setOnClickListener { cameraEngine.toggleTorch() }
        binding.gridButton.setOnClickListener {
            gridEnabled = !gridEnabled
            binding.gridButton.alpha = if (gridEnabled) 1f else 0.55f
        }
        binding.proButton.setOnClickListener { showProControls() }
        binding.flipButton.setOnClickListener {
            resetDirectorState(resetBody = true)
            cameraEngine.flip(::onCameraReady)
        }
        binding.directorToggle.setOnClickListener {
            directorEnabled = !directorEnabled
            if (!directorEnabled) {
                activePlan = null
                cameraEngine.restoreAuto()
                latestReadiness = PlanReadiness(false, 0f, "Manual camera")
                setShutterReady(false)
            }
            binding.directorToggle.text = if (directorEnabled) "AI AUTO" else "MANUAL"
            binding.directorToggle.setTextColor(getColorCompat(if (directorEnabled) R.color.director_cyan else R.color.director_muted))
        }
        binding.shutterButton.setOnClickListener { capture(manual = true) }
        binding.lensWide.setOnClickListener { cameraEngine.setZoom(0.5f) }
        binding.lensOne.setOnClickListener { cameraEngine.setZoom(1f) }
        binding.lensTele.setOnClickListener { cameraEngine.setZoom(2f) }
    }

    private fun resetDirectorState(resetBody: Boolean) {
        activePlan = null
        lastSceneSignature = ""
        lastVlmSignature = ""
        lastAppliedPlanFingerprint = ""
        latestReadiness = PlanReadiness(false, 0f, "Re-analyzing")
        cameraEngine.restoreAuto()
        if (resetBody) frameAnalyzer.resetSubjectCalibration()
        runOnUiThread { setShutterReady(false) }
    }

    private fun configureLensButtons() {
        val range = cameraEngine.zoomRange()
        binding.lensWide.visibility = if (range.start <= 0.55f) View.VISIBLE else View.GONE
        binding.lensTele.visibility = if (range.endInclusive >= 1.9f) View.VISIBLE else View.GONE
    }

    private fun onScene(scene: SceneState) {
        extensionNames = cameraEngine.availableExtensionNames()
        val signature = sceneSignature(scene)
        if (signature != lastSceneSignature) lastSceneSignature = signature

        if (directorEnabled) maybeRequestPhotoPlan(scene, signature)
        val plan = activePlan
        latestReadiness = if (directorEnabled) readinessEvaluator.evaluate(scene, plan) else PlanReadiness(false, 0f, "Manual camera")

        val frameState = DirectorFrameState(
            scene = scene,
            gridEnabled = gridEnabled,
            message = if (plan == null) "READING SCENE" else plan.theme.uppercase(),
            hint = latestReadiness.reason
        )

        runOnUiThread {
            if (directorEnabled && plan != null) applyPlanIfChanged(plan)
            binding.directorOverlay.setState(frameState, cameraEngine.isFrontCamera)
            binding.directorOverlay.setPhotoPlan(if (directorEnabled) plan else null, latestReadiness)
            renderStatus(plan, latestReadiness)
            setShutterReady(directorEnabled && latestReadiness.ready)
        }
    }

    private fun maybeRequestPhotoPlan(scene: SceneState, signature: String) {
        val caps = cameraEngine.capabilities ?: return
        if (!photoDirector.ready || scene.analysisFrameJpeg == null) return
        val now = scene.timestampMs
        val changed = signature != lastVlmSignature
        if (!changed && now - lastVlmRequestAt < VLM_REFRESH_MS) return
        if (!vlmInFlight.compareAndSet(false, true)) return

        lastVlmRequestAt = now
        lastVlmSignature = signature
        vlmExecutor.execute {
            try {
                val proposed = photoDirector.propose(scene, caps, extensionNames)
                val validated = planValidator.validate(proposed, scene, caps, extensionNames)
                if (validated != null) {
                    val previous = activePlan
                    activePlan = if (previous == null || changed) validated else previous.blendedWith(validated, PLAN_SMOOTHING)
                }
            } finally {
                vlmInFlight.set(false)
            }
        }
    }

    private fun applyPlanIfChanged(plan: PhotoPlan) {
        val fp = plan.fingerprint()
        if (fp == lastAppliedPlanFingerprint) return
        cameraEngine.applyPhotoPlan(plan)
        lastAppliedPlanFingerprint = fp
    }

    private fun renderStatus(plan: PhotoPlan?, readiness: PlanReadiness) {
        if (!vision.ready) {
            binding.directorState.text = "INITIALIZING LOCAL AI"
            binding.directorHint.text = vision.error ?: "Camera remains available"
            return
        }
        if (!directorEnabled) {
            binding.directorState.text = "MANUAL CAMERA"
            binding.directorHint.text = "Tap anywhere to focus"
            return
        }
        if (plan == null) {
            binding.directorState.text = if (photoDirector.ready) "READING SCENE" else "AI FALLBACK"
            binding.directorHint.text = photoDirector.error ?: "Analyzing light, subject, background and camera"
            return
        }

        binding.directorState.text = if (readiness.ready) "READY · ${plan.theme.uppercase()}" else plan.theme.uppercase()
        val treatment = buildList {
            add(plan.program.name)
            if (plan.extensionPreference != ExtensionPreference.NONE) add(plan.extensionPreference.name)
            add("${"%.1f".format(plan.zoomRatio)}×")
            if (plan.colorStyle.name != "NATURAL") add(plan.colorStyle.name.replace('_', ' '))
        }.joinToString(" · ")
        binding.directorHint.text = if (readiness.ready) treatment else (plan.cameraGuidance ?: plan.subjectGuidance ?: readiness.reason)
    }

    private fun setShutterReady(ready: Boolean) {
        binding.shutterButton.setBackgroundResource(if (ready) R.drawable.bg_shutter_ready else R.drawable.bg_shutter)
        binding.shutterButton.contentDescription = if (ready) "AI shot ready" else "Capture"
    }

    private fun sceneSignature(scene: SceneState): String = buildString {
        append(scene.personCount).append('|')
        append(scene.primaryBodyProfile()?.crop?.name ?: "NONE").append('|')
        scene.entities.filter { it.score >= 0.42f }.map { it.label }.distinct().sorted().take(10).forEach { append(it).append(',') }
        append('|').append((scene.quality.meanLuma * 12).toInt())
        append('|').append((scene.quality.contrast * 8).toInt())
        append('|').append((scene.quality.highlightClipping * 10).toInt())
    }

    private fun capture(manual: Boolean) {
        if (!cameraReady) {
            Toast.makeText(this, "Camera is still initializing.", Toast.LENGTH_SHORT).show()
            return
        }
        val plan = activePlan
        val wantsExtension = directorEnabled && plan != null && plan.extensionPreference != ExtensionPreference.NONE
        val extensionActive = wantsExtension && cameraEngine.prepareCaptureExtension(plan!!.extensionPreference)

        binding.directorState.text = "CAPTURING"
        binding.directorHint.text = when {
            extensionActive -> "${plan!!.theme} · OEM ${plan.extensionPreference.name.lowercase()} pipeline"
            directorEnabled && plan != null -> "${plan.theme} · Director camera treatment"
            else -> "Maximum real native-resolution capture"
        }
        setShutterReady(false)

        captureController.capture(
            count = if (extensionActive) 1 else if (directorEnabled && plan != null) plan.burstCount else 1,
            externalOutput = externalOutputUri
        ) { result ->
            runOnUiThread {
                if (extensionActive) {
                    cameraEngine.restoreAnalysisSession()
                    lastAppliedPlanFingerprint = ""
                    plan?.let { applyPlanIfChanged(it) }
                }
                result.onSuccess { uri ->
                    binding.directorState.text = "SAVED"
                    binding.directorHint.text = when {
                        extensionActive -> "OEM computational frame saved locally"
                        native4KAvailable -> "Best native 4K+ frame stored locally"
                        else -> "Best native camera frame stored locally"
                    }
                    if (intent.action == MediaStore.ACTION_IMAGE_CAPTURE) {
                        setResult(Activity.RESULT_OK, Intent().setData(uri))
                        finish()
                    }
                }.onFailure {
                    binding.directorState.text = "CAPTURE ERROR"
                    binding.directorHint.text = it.message ?: "Unable to save native frame"
                }
            }
        }
    }

    private fun showProControls() {
        val caps = cameraEngine.capabilities ?: return
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(10))
        }
        fun label(text: String): TextView = TextView(this).apply {
            this.text = text
            setTextColor(getColorCompat(R.color.director_white))
            textSize = 13f
            setPadding(0, dp(12), 0, dp(4))
        }

        panel.addView(label("AI AUTO normally controls these values. Manual changes temporarily override the active treatment."))
        val exposureRange = cameraEngine.exposureRange()
        panel.addView(label("Exposure compensation  ${exposureRange.first} … ${exposureRange.last}"))
        panel.addView(SeekBar(this).apply {
            max = max(0, exposureRange.last - exposureRange.first)
            progress = (0 - exposureRange.first).coerceIn(0, max)
            setOnSeekBarChangeListener(simpleSeek { p -> cameraEngine.setExposureCompensation(exposureRange.first + p) })
        })

        panel.addView(Switch(this).apply {
            text = "Lock white balance"
            setTextColor(getColorCompat(R.color.director_white))
            setOnCheckedChangeListener { _, checked -> cameraEngine.setAwbLock(checked) }
        })

        if (caps.supportsManualSensor && caps.sensitivityRange != null && caps.exposureTimeRange != null) {
            var currentIso = geometricInt(caps.sensitivityRange.lower, caps.sensitivityRange.upper, 0.45f)
            var currentExposure = geometricLong(caps.exposureTimeRange.lower, caps.exposureTimeRange.upper, 0.42f)
            var currentFocus: Float? = null
            val isoLabel = label("ISO  $currentIso")
            panel.addView(isoLabel)
            panel.addView(SeekBar(this).apply {
                max = 1000; progress = 450
                setOnSeekBarChangeListener(simpleSeek { p ->
                    currentIso = geometricInt(caps.sensitivityRange.lower, caps.sensitivityRange.upper, p / 1000f)
                    isoLabel.text = "ISO  $currentIso"
                    cameraEngine.applyManualSensor(currentIso, currentExposure, currentFocus)
                })
            })
            val shutterLabel = label("Shutter  ${formatExposure(currentExposure)}")
            panel.addView(shutterLabel)
            panel.addView(SeekBar(this).apply {
                max = 1000; progress = 420
                setOnSeekBarChangeListener(simpleSeek { p ->
                    currentExposure = geometricLong(caps.exposureTimeRange.lower, caps.exposureTimeRange.upper, p / 1000f)
                    shutterLabel.text = "Shutter  ${formatExposure(currentExposure)}"
                    cameraEngine.applyManualSensor(currentIso, currentExposure, currentFocus)
                })
            })
            (caps.minFocusDistance ?: 0f).takeIf { it > 0f }?.let { maxFocus ->
                val focusLabel = label("Manual focus  AUTO")
                panel.addView(focusLabel)
                panel.addView(SeekBar(this).apply {
                    max = 1000
                    setOnSeekBarChangeListener(simpleSeek { p ->
                        currentFocus = if (p == 0) null else maxFocus * p / 1000f
                        focusLabel.text = if (p == 0) "Manual focus  AUTO" else "Manual focus  ${"%.2f".format(currentFocus)} D"
                        if (currentFocus == null) cameraEngine.restoreAuto() else cameraEngine.applyManualSensor(currentIso, currentExposure, currentFocus)
                    })
                })
            }
        }

        MaterialAlertDialogBuilder(this)
            .setTitle("PRO SENSOR CONTROL")
            .setView(panel)
            .setPositiveButton("DONE", null)
            .setNeutralButton("AUTO") { _, _ -> cameraEngine.restoreAuto() }
            .show()
    }

    private fun simpleSeek(block: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) { if (fromUser) block(progress) }
        override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
        override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
    }

    private fun geometricInt(low: Int, high: Int, t: Float): Int = exp(ln(low.toDouble()) + (ln(high.toDouble()) - ln(low.toDouble())) * t).toInt()
    private fun geometricLong(low: Long, high: Long, t: Float): Long = exp(ln(low.toDouble()) + (ln(high.toDouble()) - ln(low.toDouble())) * t).toLong()
    private fun formatExposure(ns: Long): String {
        val seconds = ns / 1_000_000_000.0
        return if (seconds >= 1.0) "${"%.2f".format(seconds)}s" else "1/${max(1, (1.0 / seconds).toInt())}s"
    }

    private inner class CameraTouchHandler : View.OnTouchListener {
        private val scale = ScaleGestureDetector(this@MainActivity, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                cameraEngine.zoomBy(detector.scaleFactor)
                return true
            }
        })
        private val gesture = GestureDetector(this@MainActivity, object : GestureDetector.SimpleOnGestureListener() {
            override fun onSingleTapUp(e: MotionEvent): Boolean {
                cameraEngine.focusAt(e.x, e.y)
                return true
            }
        })
        override fun onTouch(v: View?, event: MotionEvent): Boolean {
            scale.onTouchEvent(event)
            gesture.onTouchEvent(event)
            return true
        }
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
    private fun getColorCompat(id: Int): Int = ContextCompat.getColor(this, id)

    override fun onDestroy() {
        thermalGovernor.stop()
        sensorFusion.stop()
        cameraEngine.close()
        vision.close()
        photoDirector.close()
        analysisExecutor.shutdownNow()
        captureExecutor.shutdownNow()
        vlmExecutor.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val VLM_REFRESH_MS = 1_200L
        private const val PLAN_SMOOTHING = 0.34f
    }
}
