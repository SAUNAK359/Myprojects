package com.saunak.director

import android.view.Surface
import com.saunak.director.camera.CameraEngine
import org.junit.Assert.assertEquals
import org.junit.Test

class CameraEngineMetadataTest {
    @Test fun surfaceRotationIsConvertedToDegrees() {
        assertEquals(0, CameraEngine.surfaceRotationToDegrees(Surface.ROTATION_0))
        assertEquals(90, CameraEngine.surfaceRotationToDegrees(Surface.ROTATION_90))
        assertEquals(180, CameraEngine.surfaceRotationToDegrees(Surface.ROTATION_180))
        assertEquals(270, CameraEngine.surfaceRotationToDegrees(Surface.ROTATION_270))
    }
}
