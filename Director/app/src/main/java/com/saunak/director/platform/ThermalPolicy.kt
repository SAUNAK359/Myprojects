package com.saunak.director.platform

import android.content.Context
import android.os.Build
import android.os.PowerManager

/**
 * maxPhysicalCaptures is the legacy whole-photo retry budget.
 * maxSierraBurstFrames is the internal multi-frame computational-capture budget.
 * Keeping them separate prevents a retry policy from accidentally degrading burst reconstruction.
 */
data class ThermalPolicy(
    val allowAi: Boolean,
    val maxPhysicalCaptures: Int,
    val maxAiDimension: Int,
    val maxSierraBurstFrames: Int = 8,
    val thermalPressure: Double = 0.0
) {
    init {
        require(maxPhysicalCaptures >= 1)
        require(maxAiDimension >= 1)
        require(maxSierraBurstFrames in 1..16)
        require(thermalPressure in 0.0..1.0)
    }
}

class ThermalMonitor(context: Context) {
    private val power = context.getSystemService(PowerManager::class.java)

    fun policy(): ThermalPolicy {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return ThermalPolicy(true, 3, 768, maxSierraBurstFrames = 6, thermalPressure = 0.25)
        }
        return when (power.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_CRITICAL,
            PowerManager.THERMAL_STATUS_EMERGENCY,
            PowerManager.THERMAL_STATUS_SHUTDOWN ->
                ThermalPolicy(false, 1, 448, maxSierraBurstFrames = 1, thermalPressure = 1.0)

            PowerManager.THERMAL_STATUS_SEVERE ->
                ThermalPolicy(true, 1, 512, maxSierraBurstFrames = 3, thermalPressure = 0.90)

            PowerManager.THERMAL_STATUS_MODERATE ->
                ThermalPolicy(true, 2, 640, maxSierraBurstFrames = 5, thermalPressure = 0.62)

            PowerManager.THERMAL_STATUS_LIGHT ->
                ThermalPolicy(true, 3, 704, maxSierraBurstFrames = 6, thermalPressure = 0.35)

            else ->
                ThermalPolicy(true, 3, 768, maxSierraBurstFrames = 8, thermalPressure = 0.08)
        }
    }
}
