package ai.auraforge.director.core

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GeometryTest {
    @Test fun iouIsOneForIdenticalBoxes() {
        val a = RectF(0.1f, 0.2f, 0.7f, 0.9f)
        assertEquals(1f, Geometry.intersectionOverUnion(a, a), 0.0001f)
    }

    @Test fun bboxContainsVisiblePoints() {
        val points = listOf(Point2(0.2f, 0.3f), Point2(0.8f, 0.7f), Point2(0.4f, 0.5f))
        val box = Geometry.bbox(points, 0f)
        assertTrue(box.left <= 0.2f && box.right >= 0.8f)
        assertTrue(box.top <= 0.3f && box.bottom >= 0.7f)
    }
}
