package com.saunak.director

import com.saunak.director.engine.geometry.CoordinateGraph
import com.saunak.director.engine.geometry.CoordinateSpace
import com.saunak.director.engine.geometry.Point2D
import com.saunak.director.engine.geometry.Transform3x3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class CoordinateGraphTest {
    @Test fun composesAndInvertsCoordinatePaths() {
        val graph = CoordinateGraph()
        graph.connect(CoordinateSpace.SENSOR_RAW, CoordinateSpace.ACTIVE_ARRAY, Transform3x3.translate(10.0, 20.0))
        graph.connect(CoordinateSpace.ACTIVE_ARRAY, CoordinateSpace.MODEL_INPUT, Transform3x3.scale(0.5, 0.25))

        val mapped = graph.map(Point2D(100.0, 80.0), CoordinateSpace.SENSOR_RAW, CoordinateSpace.MODEL_INPUT)
        assertEquals(55.0, mapped.x, 1e-6)
        assertEquals(25.0, mapped.y, 1e-6)

        val back = graph.map(mapped, CoordinateSpace.MODEL_INPUT, CoordinateSpace.SENSOR_RAW)
        assertEquals(100.0, back.x, 1e-6)
        assertEquals(80.0, back.y, 1e-6)
        assertNotNull(graph.transform(CoordinateSpace.MODEL_INPUT, CoordinateSpace.SENSOR_RAW))
    }
}
