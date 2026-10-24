package com.saunak.director.engine.reconstruction

import com.saunak.director.camera.CameraEngine
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.RandomAccessFile

/** Metadata for the compact Sierra RAW_SENSOR container written directly from ImageProxy. */
data class SierraRawHeader(
    val width: Int,
    val height: Int,
    val rowStrideBytes: Int,
    val pixelStrideBytes: Int,
    val rotationDegrees: Int,
    val exposureTimeNs: Long,
    val sensitivityIso: Int,
    val sensorTimestampNs: Long,
    val payloadBytes: Int,
    val payloadOffset: Long
) {
    init {
        require(width > 0 && height > 0)
        require(rowStrideBytes > 0)
        require(pixelStrideBytes >= 2)
        require(rotationDegrees in setOf(0, 90, 180, 270))
        require(exposureTimeNs > 0)
        require(sensitivityIso > 0)
        require(payloadBytes > 0)
        require(payloadOffset > 0)
    }
}

/**
 * Random-access 16-bit RAW sensor view. It never maps/copies the whole sensor image into Java heap.
 * Android RAW_SENSOR samples are interpreted as little-endian 16-bit containers with the actual
 * sensor bit depth represented by white/black calibration values from DeviceTwin.
 */
class SierraRawContainer private constructor(
    val file: File,
    val header: SierraRawHeader,
    private val raf: RandomAccessFile
) : AutoCloseable {

    fun sampleUnsigned16(x: Int, y: Int): Int {
        require(x in 0 until header.width && y in 0 until header.height)
        val offset = header.payloadOffset + y.toLong() * header.rowStrideBytes + x.toLong() * header.pixelStrideBytes
        check(offset + 1 < header.payloadOffset + header.payloadBytes) { "RAW sample exceeds payload" }
        raf.seek(offset)
        val lo = raf.readUnsignedByte()
        val hi = raf.readUnsignedByte()
        return lo or (hi shl 8)
    }

    /** Reads a rectangular region into a compact primitive array for tile reconstruction. */
    fun readTile(x0: Int, y0: Int, width: Int, height: Int): IntArray {
        require(x0 >= 0 && y0 >= 0 && width > 0 && height > 0)
        require(x0 + width <= header.width && y0 + height <= header.height)
        val out = IntArray(width * height)
        var index = 0
        for (y in y0 until y0 + height) {
            for (x in x0 until x0 + width) out[index++] = sampleUnsigned16(x, y)
        }
        return out
    }

    override fun close() = raf.close()

    companion object {
        fun open(file: File): SierraRawContainer {
            require(file.isFile && file.length() > 0) { "Missing Sierra RAW file" }
            val header = readHeader(file)
            val raf = RandomAccessFile(file, "r")
            check(file.length() >= header.payloadOffset + header.payloadBytes) { "Truncated Sierra RAW payload" }
            return SierraRawContainer(file, header, raf)
        }

        private fun readHeader(file: File): SierraRawHeader {
            DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
                check(input.readUTF() == CameraEngine.SRAW_MAGIC) { "Not a Sierra RAW container" }
                check(input.readInt() == CameraEngine.SRAW_VERSION) { "Unsupported Sierra RAW version" }
                val width = input.readInt()
                val height = input.readInt()
                val rowStride = input.readInt()
                val pixelStride = input.readInt()
                val rotation = input.readInt()
                val exposure = input.readLong()
                val iso = input.readInt()
                val timestamp = input.readLong()
                val payloadBytes = input.readInt()
                // DataInputStream has consumed exactly this many bytes; use file pointer from an
                // independent RandomAccessFile to avoid relying on buffering internals.
                RandomAccessFile(file, "r").use { probe ->
                    probe.readUTF()
                    probe.readInt()
                    repeat(5) { probe.readInt() }
                    probe.readLong()
                    probe.readInt()
                    probe.readLong()
                    probe.readInt()
                    return SierraRawHeader(
                        width = width,
                        height = height,
                        rowStrideBytes = rowStride,
                        pixelStrideBytes = pixelStride,
                        rotationDegrees = rotation,
                        exposureTimeNs = exposure,
                        sensitivityIso = iso,
                        sensorTimestampNs = timestamp,
                        payloadBytes = payloadBytes,
                        payloadOffset = probe.filePointer
                    )
                }
            }
        }
    }
}
