package ai.auraforge.director.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.abs
import kotlin.math.sqrt

class SensorFusion(context: Context) : SensorEventListener {
    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro = manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val rotation = manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    @Volatile var motionScore: Float = 0f
        private set
    @Volatile var rollDeg: Float = 0f
        private set

    private var filteredMotion = 0f

    fun start() {
        gyro?.let { manager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        rotation?.let { manager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() = manager.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                val m = sqrt(event.values[0] * event.values[0] + event.values[1] * event.values[1] + event.values[2] * event.values[2])
                val normalized = (m / 2.6f).coerceIn(0f, 1f)
                filteredMotion = 0.72f * filteredMotion + 0.28f * normalized
                motionScore = filteredMotion
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                val rotationMatrix = FloatArray(9)
                val orientation = FloatArray(3)
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                SensorManager.getOrientation(rotationMatrix, orientation)
                rollDeg = Math.toDegrees(orientation[2].toDouble()).toFloat()
                if (abs(rollDeg) > 90f) rollDeg = 0f
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
