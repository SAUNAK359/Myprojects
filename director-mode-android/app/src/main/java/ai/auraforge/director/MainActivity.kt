package ai.auraforge.director

import ai.auraforge.director.camera.CameraCapabilities
import ai.auraforge.director.camera.CameraEngine
import ai.auraforge.director.capture.BurstCaptureController
import ai.auraforge.director.core.DirectorFrameState
import ai.auraforge.director.core.DirectorLockState
import ai.auraforge.director.core.SceneState
import ai.auraforge.director.core.ThermalGovernor
import ai.auraforge.director.databinding.ActivityMainBinding
import ai.auraforge.director.director.AlignmentEngine
import ai.auraforge.director.director.DirectorBrain
import ai.auraforge.director.director.DirectorConfig
import ai.auraforge.director.director.ShotTemplateRepository
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
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var captureExecutor: ExecutorService

    private lateinit var sensorFusion: SensorFusion
    private lateinit var vision: VisionPipeline
    private lateinit var imageQuality: ImageQualityAnalyzer
    private lateinit var frameAnalyzer: FrameAnalyzer
    private lateinit var cameraEngine: CameraEngine
    private lateinit var captureController: BurstCaptureController

    private lateinit var directorConfig: DirectorConfig
    private lateinit var directorBrain: DirectorBrain
    private lateinit var alignmentEngine: AlignmentEngine
    private lateinit var thermalGovernor: ThermalGovernor

    private var directorEnabled = true
    private var gridEnabled = false
    private var native4KAvailable = false
    private var cameraReady = false
    private var lastState = DirectorFrameState()
    private var lastSceneSignature = ""
    private var externalOutputUri: Uri? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
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

        directorConfig = DirectorConfig.load(this)
        val templates = ShotTemplateRepository.load(this)
        directorBrain = DirectorBrain(templates, directorConfig)
        alignmentEngine = AlignmentEngine(directorConfig)
        sensorFusion = SensorFusion(this)
        imageQuality = ImageQualityAnalyzer()

        analysisExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "Director-Analysis").apply { priority = Thread.NORM_PRIORITY + 1 } }
        captureExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "Director-Capture") }
        vision = VisionPipeline(this, preferGpu = true)
        frameAnalyzer = FrameAnalyzer(vision, imageQuality, sensorFusion, ::onScene)

        cameraEngine = CameraEngine(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            analyzer = frameAnalyzer,
            analysisExecutor = analysisExecutor
        )

        captureController = BurstCaptureController(
            context = this,
            imageCaptureProvider = { cameraEngine.imageCapture },
            executor = captureExecutor,
            qualityAnalyzer = imageQuality
        )

        thermalGovernor = ThermalGovernor(this) { level, label, interval ->
            frameAnalyzer.targetIntervalMs = interval
            runOnUiThread {
                binding.thermalBadge.text = "● $label"
                binding.thermalBadge.setTextColor(
                    when {
                        level >= 3 -> getColorCompat(R.color.director_danger)
                        level >= 1 -> getColorCompat(R.color.director_amber)
                        else -> getColorCompat(R.color.director_green)
                    }
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
        cameraEngine.start(
            onReady = ::onCameraReady,
            onError = { t -> runOnUiThread { Toast.makeText(this, "Camera failed: ${t.message}", Toast.LENGTH_LONG).show() } }
        )
    }

    private fun onCameraReady(caps: CameraCapabilities) {
        cameraReady = true
        native4KAvailable = caps.native4K
        val size = caps.maxJpegSize
        runOnUiThread {
            binding.resolutionBadge.text = if (caps.native4K) {
                "NATIVE 4K+"
            } else {
                size?.let { "MAX ${it.width}×${it.height}" } ?: "MAX SENSOR"
            }
            binding.resolutionBadge.setTextColor(getColorCompat(if (caps.native4K) R.color.director_cyan else R.color.director_amber))
            configureLensButtons()
            if (!caps.native4K) {
                binding.directorHint.text = "Preview works; capture requires a native ≥3840×2160 sensor output"
            }
        }
    }

    private fun setupUi() {
        binding.previewView.setOnTouchListener(CameraTouchHandler())

        binding.torchButton.setOnClickListener { cameraEngine.toggleTorch() }
        binding.gridButton.setOnClickListener {
            gridEnabled = !gridEnabled
            binding.gridButton.alpha = if (gridEnabled) 1f else 0.55f
            binding.directorOverlay.setState(lastState.copy(gridEnabled = gridEnabled), cameraEngine.isFrontCamera)
        }
        binding.proButton.setOnClickListener { showProControls() }
        binding.flipButton.setOnClickListener {
            directorBrain.releasePlan()
            alignmentEngine.reset()
            cameraEngine.flip(::onCameraReady)
        }
        binding.directorToggle.setOnClickListener {
            directorEnabled = !directorEnabled
            if (!directorEnabled) {
                directorBrain.releasePlan()
                alignmentEngine.reset()
            }
            binding.directorToggle.text = if (directorEnabled) "DIRECTOR ON" else "DIRECTOR OFF"
            binding.directorToggle.setTextColor(getColorCompat(if (directorEnabled) R.color.director_cyan else R.color.director_muted))
        }
        binding.shutterButton.setOnClickListener { capture(manual = true) }

        binding.lensWide.setOnClickListener { cameraEngine.setZoom(0.5f) }
        binding.lensOne.setOnClickListener { cameraEngine.setZoom(1f) }
        binding.lensTele.setOnClickListener { cameraEngine.setZoom(2f) }
    }

    private fun configureLensButtons() {
        val range = cameraEngine.zoomRange()
        binding.lensWide.visibility = if (range.start <= 0.55f) View.VISIBLE else View.GONE
        binding.lensTele.visibility = if (range.endInclusive >= 1.9f) View.VISIBLE else View.GONE
    }

    private fun onScene(scene: SceneState) {
        val signature = buildString {
            append(scene.personCount)
            append('|')
            scene.entities.map { it.label }.distinct().sorted().take(8).forEach { append(it).append(',') }
        }
        if (signature != lastSceneSignature) {
            lastSceneSignature = signature
            directorBrain.releasePlan()
            alignmentEngine.reset()
        }

        val plan = if (directorEnabled) directorBrain.plan(scene) else null
        val alignment = alignmentEngine.evaluate(scene, plan)
        val (message, hint) = statusText(scene, plan?.displayName, alignment.state)
        val frameState = DirectorFrameState(scene, plan, alignment, gridEnabled, message, hint)
        lastState = frameState

        runOnUiThread {
            binding.directorOverlay.setState(frameState, cameraEngine.isFrontCamera)
            binding.directorState.text = message
            binding.directorHint.text = hint
            if (alignment.state == DirectorLockState.LOCKED && alignmentEngine.canCaptureAgain(scene.timestampMs)) {
                capture(manual = false)
            }
        }
    }

    private fun statusText(scene: SceneState, planName: String?, state: DirectorLockState): Pair<String, String> {
        if (!vision.ready) return "INITIALIZING LOCAL AI" to (vision.error ?: "Models stay entirely on device")
        if (scene.poses.isEmpty()) return "FINDING SUBJECT" to "Frame a person to start Director Mode"
        if (!directorEnabled) return "MANUAL CAMERA" to "Director staging is paused"
        if (planName == null) return "AUTO COMPOSITION" to "Current setup is already usable — keep framing"

        return when (state) {
            DirectorLockState.SEARCHING, DirectorLockState.SHOT_SELECTED -> "DESIGNING SHOT" to planName
            DirectorLockState.ALIGNING -> planName.uppercase() to "Align your body with the ghost frame"
            DirectorLockState.NEAR_LOCK -> "ALMOST LOCKED" to "Small adjustment — keep matching the guide"
            DirectorLockState.LOCK_CANDIDATE -> "HOLD" to "Pose matched — stabilizing focus and moment"
            DirectorLockState.LOCKED -> "LOCKED" to "Capturing the strongest frame"
            DirectorLockState.CAPTURING -> "CAPTURING" to "Selecting the sharpest frame locally"
        }
    }

    private fun capture(manual: Boolean) {
        if (!cameraReady) {
            Toast.makeText(this, "Camera is still initializing.", Toast.LENGTH_SHORT).show()
            return
        }
        if (!native4KAvailable) {
            Toast.makeText(this, "This camera does not expose a native 4K still output. Director will not upscale or fake 4K.", Toast.LENGTH_LONG).show()
            return
        }
        val now = System.nanoTime() / 1_000_000L
        alignmentEngine.markCaptured(now)
        binding.directorState.text = "CAPTURING"
        binding.directorHint.text = if (manual) "Native full-resolution capture" else "Pose locked — ranking a micro-burst"

        captureController.capture(
            count = if (manual) 1 else 3,
            externalOutput = externalOutputUri
        ) { result ->
            runOnUiThread {
                result.onSuccess { uri ->
                    binding.directorState.text = "SAVED"
                    binding.directorHint.text = "Best native frame stored locally"
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

        val exposureRange = cameraEngine.exposureRange()
        panel.addView(label("Exposure compensation  ${exposureRange.first} … ${exposureRange.last}"))
        panel.addView(SeekBar(this).apply {
            max = max(0, exposureRange.last - exposureRange.first)
            progress = (0 - exposureRange.first).coerceIn(0, max)
            setOnSeekBarChangeListener(simpleSeek { p -> cameraEngine.setExposureCompensation(exposureRange.first + p) })
        })

        val awb = Switch(this).apply {
            text = "Lock white balance"
            setTextColor(getColorCompat(R.color.director_white))
            setOnCheckedChangeListener { _, checked -> cameraEngine.setAwbLock(checked) }
        }
        panel.addView(awb)

        if (caps.supportsManualSensor && caps.sensitivityRange != null && caps.exposureTimeRange != null) {
            var currentIso = geometricInt(caps.sensitivityRange.lower, caps.sensitivityRange.upper, 0.45f)
            var currentExposure = geometricLong(caps.exposureTimeRange.lower, caps.exposureTimeRange.upper, 0.42f)
            var currentFocus: Float? = null

            val isoLabel = label("ISO  $currentIso")
            panel.addView(isoLabel)
            panel.addView(SeekBar(this).apply {
                max = 1000
                progress = 450
                setOnSeekBarChangeListener(simpleSeek { p ->
                    currentIso = geometricInt(caps.sensitivityRange.lower, caps.sensitivityRange.upper, p / 1000f)
                    isoLabel.text = "ISO  $currentIso"
                    cameraEngine.applyManualSensor(currentIso, currentExposure, currentFocus)
                })
            })

            val shutterLabel = label("Shutter  ${formatExposure(currentExposure)}")
            panel.addView(shutterLabel)
            panel.addView(SeekBar(this).apply {
                max = 1000
                progress = 420
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
                    progress = 0
                    setOnSeekBarChangeListener(simpleSeek { p ->
                        currentFocus = if (p == 0) null else maxFocus * p / 1000f
                        focusLabel.text = if (p == 0) "Manual focus  AUTO" else "Manual focus  ${"%.2f".format(currentFocus)} D"
                        if (currentFocus == null) cameraEngine.restoreAuto() else cameraEngine.applyManualSensor(currentIso, currentExposure, currentFocus)
                    })
                })
            }
        } else {
            panel.addView(label("Manual ISO/shutter is not exposed by this camera HAL. Auto/EV/focus/zoom remain available."))
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
        analysisExecutor.shutdownNow()
        captureExecutor.shutdownNow()
        super.onDestroy()
    }
}
