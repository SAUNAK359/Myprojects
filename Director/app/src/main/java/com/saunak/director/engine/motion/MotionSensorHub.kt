package com.saunak.director.engine.motion

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import java.util.ArrayDeque
import kotlin.math.sqrt

/**
 * Application-scoped, low-overhead IMU sampler. SensorEvent timestamps are retained in their native
 * elapsed-realtime time base so camera timestamps can be fused only when the camera HAL declares
 * SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME.
 */
class MotionSensorHub(context: Context) : AutoCloseable, SensorEventListener {
    private data class VecSample(val tNs: Long, val x: Double, val y: Double, val z: Double)

    private val manager = context.applicationContext.getSystemService(SensorManager::class.java)
    private val gyro = manager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accel = manager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val thread = HandlerThread("sierra-motion-imu").apply { start() }
    private val handler = Handler(thread.looper)
    private val lock = Any()
    private val gyroSamples = ArrayDeque<VecSample>()
    private val accelSamples = ArrayDeque<VecSample>()

    @Volatile private var running = false

    fun start() {
        if (running) return
        running = true
        gyro?.let { manager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME, handler) }
        accel?.let { manager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME, handler) }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!running || event.values.size < 3 || event.timestamp <= 0L) return
        val sample = VecSample(
            tNs = event.timestamp,
            x = event.values[0].toDouble(),
            y = event.values[1].toDouble(),
            z = event.values[2].toDouble()
        )
        synchronized(lock) {
            when (event.sensor.type) {
                Sensor.TYPE_GYROSCOPE -> gyroSamples.addLast(sample)
                Sensor.TYPE_ACCELEROMETER -> accelSamples.addLast(sample)
            }
            trimLocked(event.timestamp)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /**
     * Summarizes the IMU around a frame timestamp. When timestamps are not comparable, the caller
     * still receives fresh device-motion evidence but it is explicitly marked unaligned and is
     * down-weighted by MotionFusionEngine.
     */
    fun observation(
        frameSensorTimestampNs: Long,
        frameCallbackElapsedRealtimeNs: Long,
        timestampComparable: Boolean,
        windowNs: Long = DEFAULT_WINDOW_NS
    ): ImuMotionObservation {
        val center = if (timestampComparable) frameSensorTimestampNs else frameCallbackElapsedRealtimeNs
        val start = center - windowNs
        val end = center + windowNs / 4
        val gyroWindow: List<VecSample>
        val accelWindow: List<VecSample>
        synchronized(lock) {
            gyroWindow = gyroSamples.filter { it.tNs in start..end }
            accelWindow = accelSamples.filter { it.tNs in start..end }
        }

        val angular = if (gyroWindow.isEmpty()) 0.0 else gyroWindow.map(::magnitude).average()
        val accelMagnitudes = accelWindow.map(::magnitude)
        val accelStd = if (accelMagnitudes.size < 2) 0.0 else {
            val mean = accelMagnitudes.average()
            sqrt(accelMagnitudes.sumOf { (it - mean) * (it - mean) } / accelMagnitudes.size)
        }
        val coverage = (
            (gyroWindow.size / EXPECTED_GYRO_SAMPLES.toDouble()).coerceIn(0.0, 1.0) * 0.72 +
                (accelWindow.size / EXPECTED_ACCEL_SAMPLES.toDouble()).coerceIn(0.0, 1.0) * 0.28
            ).coerceIn(0.0, 1.0)
        val alignmentFactor = if (timestampComparable) 1.0 else 0.55

        return ImuMotionObservation(
            angularVelocityRadPerSec = angular,
            linearAccelerationStdMps2 = accelStd,
            confidence = (coverage * alignmentFactor).coerceIn(0.0, 1.0),
            observedAtElapsedRealtimeNs = frameCallbackElapsedRealtimeNs.coerceAtLeast(SystemClock.elapsedRealtimeNanos() - 2_000_000_000L),
            timestampAligned = timestampComparable,
            gyroSamples = gyroWindow.size,
            accelerationSamples = accelWindow.size
        )
    }

    private fun trimLocked(nowNs: Long) {
        val cutoff = nowNs - RETENTION_NS
        while (gyroSamples.firstOrNull()?.tNs?.let { it < cutoff } == true) gyroSamples.removeFirst()
        while (accelSamples.firstOrNull()?.tNs?.let { it < cutoff } == true) accelSamples.removeFirst()
        while (gyroSamples.size > MAX_SAMPLES) gyroSamples.removeFirst()
        while (accelSamples.size > MAX_SAMPLES) accelSamples.removeFirst()
    }

    private fun magnitude(sample: VecSample): Double = sqrt(sample.x * sample.x + sample.y * sample.y + sample.z * sample.z)

    override fun close() {
        running = false
        manager?.unregisterListener(this)
        synchronized(lock) {
            gyroSamples.clear()
            accelSamples.clear()
        }
        thread.quitSafely()
    }

    companion object {
        private const val DEFAULT_WINDOW_NS = 120_000_000L
        private const val RETENTION_NS = 1_500_000_000L
        private const val MAX_SAMPLES = 512
        private const val EXPECTED_GYRO_SAMPLES = 6
        private const val EXPECTED_ACCEL_SAMPLES = 6
    }
}
