package ai.auraforge.director.core

import android.content.Context
import android.os.Build
import android.os.PowerManager

class ThermalGovernor(
    context: Context,
    private val onStatus: (level: Int, label: String, analysisIntervalMs: Long) -> Unit
) {
    private val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private var listener: PowerManager.OnThermalStatusChangedListener? = null

    fun start() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            onStatus(0, "COOL", 55L)
            return
        }
        val l = PowerManager.OnThermalStatusChangedListener { status ->
            val (label, interval) = when {
                status >= PowerManager.THERMAL_STATUS_SEVERE -> "HOT" to 105L
                status >= PowerManager.THERMAL_STATUS_MODERATE -> "WARM" to 78L
                status >= PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT" to 64L
                else -> "COOL" to 55L
            }
            onStatus(status, label, interval)
        }
        listener = l
        power.addThermalStatusListener(l)
        l.onThermalStatusChanged(power.currentThermalStatus)
    }

    fun stop() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            listener?.let { power.removeThermalStatusListener(it) }
        }
        listener = null
    }
}
