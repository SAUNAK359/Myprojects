package com.saunak.director.platform

import android.content.Context
import android.os.Build
import android.os.PowerManager

data class ThermalPolicy(val allowAi: Boolean, val maxPhysicalCaptures: Int, val maxAiDimension: Int)
class ThermalMonitor(context: Context) {
    private val power = context.getSystemService(PowerManager::class.java)
    fun policy(): ThermalPolicy {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return ThermalPolicy(true, 3, 768)
        return when (power.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_CRITICAL, PowerManager.THERMAL_STATUS_EMERGENCY, PowerManager.THERMAL_STATUS_SHUTDOWN -> ThermalPolicy(false, 1, 448)
            PowerManager.THERMAL_STATUS_SEVERE -> ThermalPolicy(true, 1, 512)
            PowerManager.THERMAL_STATUS_MODERATE -> ThermalPolicy(true, 2, 640)
            else -> ThermalPolicy(true, 3, 768)
        }
    }
}
