package com.saunak.director

import com.saunak.director.engine.motion.LumaMotionFrame
import com.saunak.director.engine.motion.VisualMotionTracker
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VisualMotionTrackerTest {
    @Test fun detectsGlobalTranslationAcrossFrames() {
        val width = 112
        val height = 64
        val base = ByteArray(width * height) { index ->
            val x = index % width
            val y = index / width
            ((x * 11 + y * 17 + (x * y) % 31) and 0xff).toByte()
        }
        val shifted = ByteArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val srcX = (x - 2).coerceIn(0, width - 1)
                shifted[y * width + x] = base[y * width + srcX]
            }
        }
        val tracker = VisualMotionTracker()
        val first = LumaMotionFrame(width, height, base, 1_000_000_000L, 1_000_000_000L, 0)
        val second = LumaMotionFrame(width, height, shifted, 1_033_333_333L, 1_033_333_333L, 0)
        tracker.push(first)
        val observation = tracker.push(second)
        assertNotNull(observation)
        assertTrue(observation!!.trackCount >= 12)
        assertTrue(observation.globalVector.speedPxPerSecond > 20.0)
        assertTrue(observation.confidence > 0.25)
    }
}
