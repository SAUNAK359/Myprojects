package com.saunak.director.engine.prif

import com.saunak.director.engine.subject.NormalizedRect

/** Maps oriented analysis-frame bounds back into RAW sensor coordinates. */
internal fun orientedToSensorBounds(bounds: NormalizedRect, rotation: Int): NormalizedRect = when (rotation) {
    0 -> bounds
    90 -> NormalizedRect(bounds.top, 1.0 - bounds.right, bounds.bottom, 1.0 - bounds.left).clamped()
    180 -> NormalizedRect(
        1.0 - bounds.right,
        1.0 - bounds.bottom,
        1.0 - bounds.left,
        1.0 - bounds.top
    ).clamped()
    270 -> NormalizedRect(1.0 - bounds.bottom, bounds.left, 1.0 - bounds.top, bounds.right).clamped()
    else -> bounds
}
