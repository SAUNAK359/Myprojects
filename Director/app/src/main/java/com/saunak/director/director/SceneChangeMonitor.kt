package com.saunak.director.director

import android.graphics.Bitmap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

class SceneChangeMonitor(private val minIntervalNanos: Long = 250_000_000L, private val meanDeltaThreshold: Int = 42) {
    private val version = AtomicLong(0L)
    @Volatile private var lastObservedNanos = 0L
    @Volatile private var lastSignature: IntArray? = null
    fun currentVersion(): Long = version.get()
    @Synchronized fun observe(bitmap: Bitmap, timestampNanos: Long): Boolean {
        if (timestampNanos - lastObservedNanos < minIntervalNanos || bitmap.isRecycled) return false
        lastObservedNanos = timestampNanos
        val signature = signature(bitmap)
        val previous = lastSignature
        lastSignature = signature
        if (previous == null) return false
        val delta = signature.indices.sumOf { abs(signature[it] - previous[it]) } / signature.size
        if (delta >= meanDeltaThreshold) { version.incrementAndGet(); return true }
        return false
    }
    private fun signature(bitmap: Bitmap): IntArray {
        val luminance = IntArray(64); var index = 0; var sum = 0
        for (gy in 0 until 8) {
            val y = ((gy + 0.5f) * bitmap.height / 8f).toInt().coerceIn(0, bitmap.height - 1)
            for (gx in 0 until 8) {
                val x = ((gx + 0.5f) * bitmap.width / 8f).toInt().coerceIn(0, bitmap.width - 1)
                val c = bitmap.getPixel(x, y); val r = (c shr 16) and 0xff; val g = (c shr 8) and 0xff; val b = c and 0xff
                val yValue = (r * 77 + g * 150 + b * 29) shr 8
                luminance[index++] = yValue; sum += yValue
            }
        }
        val mean = sum / luminance.size
        return IntArray(luminance.size) { i -> luminance[i] - mean }
    }
}
