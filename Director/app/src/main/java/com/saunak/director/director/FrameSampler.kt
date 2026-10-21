package com.saunak.director.director

import android.graphics.Bitmap
import java.util.concurrent.atomic.AtomicLong

class FrameSampler(private val minIntervalMs: Long = 1500L) {
    private val lastNanos = AtomicLong(0)
    fun accept(bitmap: Bitmap, timestampNanos: Long, busy: Boolean): Bitmap? {
        if (busy) { bitmap.recycle(); return null }
        val previous = lastNanos.get()
        if (previous != 0L && timestampNanos - previous < minIntervalMs * 1_000_000L) { bitmap.recycle(); return null }
        if (bitmap.width < 64 || bitmap.height < 64) { bitmap.recycle(); return null }
        lastNanos.set(timestampNanos)
        return bitmap
    }
}
