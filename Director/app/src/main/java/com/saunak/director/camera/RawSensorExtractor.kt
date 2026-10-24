package com.saunak.director.camera

import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.params.BlackLevelPattern
import androidx.camera.core.ImageProxy
import com.saunak.director.engine.reconstruction.BayerPattern
import com.saunak.director.engine.reconstruction.CfaBlackLevel
import com.saunak.director.engine.reconstruction.RawPlane
import java.nio.ByteOrder

data class RawSensorCalibration(
    val pattern: BayerPattern,
    val whiteLevel: Float,
    val staticBlackLevel: CfaBlackLevel
) {
    init { require(whiteLevel > staticBlackLevel.offsets.max()) }
}

data class RawSensorFrame(
    val timestampNs: Long,
    val raw: RawPlane,
    val rotationDegrees: Int,
    val telemetry: CameraSensorTelemetry?
) {
    init {
        require(timestampNs > 0)
        require(rotationDegrees in setOf(0, 90, 180, 270))
    }
}

/** Copies RAW_SENSOR bytes before ImageProxy.close(), respecting arbitrary plane strides. */
object RawSensorExtractor {
    fun calibration(characteristics: CameraCharacteristics): RawSensorCalibration? {
        val cfa = characteristics.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)
            ?: return null
        val pattern = when (cfa) {
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> BayerPattern.RGGB
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> BayerPattern.BGGR
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> BayerPattern.GRBG
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> BayerPattern.GBRG
            else -> return null
        }
        val white = characteristics.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL)?.toFloat() ?: return null
        val black = characteristics.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
            ?.let(::blackLevelPattern)
            ?: CfaBlackLevel.uniform(0f)
        if (white <= black.offsets.max()) return null
        return RawSensorCalibration(pattern, white, black)
    }

    fun extract(
        image: ImageProxy,
        calibration: RawSensorCalibration,
        dynamicBlackLevel: FloatArray? = null,
        telemetry: CameraSensorTelemetry? = null
    ): RawSensorFrame {
        require(image.format == ImageFormat.RAW_SENSOR) { "Expected RAW_SENSOR ImageProxy, got ${image.format}" }
        require(image.width > 1 && image.height > 1)
        require(image.planes.size == 1) { "RAW_SENSOR should expose one packed plane" }
        val plane = image.planes[0]
        require(plane.pixelStride >= 2) { "RAW pixel stride ${plane.pixelStride} cannot hold 16-bit samples" }
        require(plane.rowStride >= image.width * plane.pixelStride)

        val buffer = plane.buffer.duplicate().order(ByteOrder.nativeOrder())
        val out = FloatArray(image.width * image.height)
        for (y in 0 until image.height) {
            val rowBase = y * plane.rowStride
            for (x in 0 until image.width) {
                val offset = rowBase + x * plane.pixelStride
                require(offset + 1 < buffer.limit()) { "RAW plane stride exceeds buffer capacity" }
                out[y * image.width + x] = (buffer.getShort(offset).toInt() and 0xffff).toFloat()
            }
        }

        val dynamic = dynamicBlackLevel
            ?.takeIf { it.size == 4 && it.all { value -> value.isFinite() && value >= 0f } }
            ?.let { CfaBlackLevel(it.copyOf()) }
            ?: calibration.staticBlackLevel

        return RawSensorFrame(
            timestampNs = image.imageInfo.timestamp,
            raw = RawPlane(
                width = image.width,
                height = image.height,
                values = out,
                pattern = calibration.pattern,
                blackLevel = dynamic.offsets.min(),
                whiteLevel = calibration.whiteLevel,
                blackLevelPattern = dynamic
            ),
            rotationDegrees = image.imageInfo.rotationDegrees,
            telemetry = telemetry
        )
    }

    private fun blackLevelPattern(pattern: BlackLevelPattern): CfaBlackLevel = CfaBlackLevel(
        floatArrayOf(
            pattern.getOffsetForIndex(0, 0).toFloat(),
            pattern.getOffsetForIndex(1, 0).toFloat(),
            pattern.getOffsetForIndex(0, 1).toFloat(),
            pattern.getOffsetForIndex(1, 1).toFloat()
        )
    )
}
