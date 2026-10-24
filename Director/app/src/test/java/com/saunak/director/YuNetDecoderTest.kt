package com.saunak.director

import com.saunak.director.engine.experts.FaceDetection
import com.saunak.director.engine.experts.FacePoint
import com.saunak.director.engine.experts.YuNetDecoder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YuNetDecoderTest {
    @Test fun decodesKnownStrideEightAnchor() {
        val width = 32
        val height = 32
        val outputs = mutableMapOf<String, FloatArray>()
        for (stride in intArrayOf(8, 16, 32)) {
            val anchors = (width / stride) * (height / stride)
            outputs["cls_$stride"] = FloatArray(anchors)
            outputs["obj_$stride"] = FloatArray(anchors)
            outputs["bbox_$stride"] = FloatArray(anchors * 4)
            outputs["kps_$stride"] = FloatArray(anchors * 10)
        }

        val anchor = 5 // row=1, col=1 for a 4x4 stride-8 grid
        outputs.getValue("cls_8")[anchor] = 1f
        outputs.getValue("obj_8")[anchor] = 1f
        val bbox = outputs.getValue("bbox_8")
        bbox[anchor * 4] = 0.5f
        bbox[anchor * 4 + 1] = 0.5f
        bbox[anchor * 4 + 2] = 0f
        bbox[anchor * 4 + 3] = 0f
        val kps = outputs.getValue("kps_8")
        repeat(5) { point ->
            kps[anchor * 10 + point * 2] = 0.5f
            kps[anchor * 10 + point * 2 + 1] = 0.5f
        }

        val faces = YuNetDecoder.decode(outputs, width, height, scoreThreshold = 0.5f)
        assertEquals(1, faces.size)
        val face = faces.single()
        assertEquals(8f, face.x, 1e-4f)
        assertEquals(8f, face.y, 1e-4f)
        assertEquals(8f, face.width, 1e-4f)
        assertEquals(8f, face.height, 1e-4f)
        assertEquals(1f, face.score, 1e-5f)
        assertEquals(12f, face.landmarks.first().x, 1e-4f)
        assertEquals(12f, face.landmarks.first().y, 1e-4f)
    }

    @Test fun nmsKeepsHigherScoringOverlappingFace() {
        val a = FaceDetection(0f, 0f, 100f, 100f, List(5) { FacePoint(20f, 20f) }, 0.95f)
        val b = FaceDetection(5f, 5f, 100f, 100f, List(5) { FacePoint(25f, 25f) }, 0.75f)
        assertTrue(YuNetDecoder.intersectionOverUnion(a, b) > 0.3f)
    }

    @Test fun mapsAndClipsDetectionBackToSourceSpace() {
        val detection = FaceDetection(-10f, 10f, 100f, 120f, List(5) { FacePoint(40f, 40f) }, 0.9f)
        val mapped = YuNetDecoder.mapToSource(detection, 640, 480, 320, 256)
        assertEquals(0f, mapped.x, 0f)
        assertTrue(mapped.y >= 0f)
        assertTrue(mapped.x + mapped.width <= 640f)
        assertTrue(mapped.y + mapped.height <= 480f)
    }
}
