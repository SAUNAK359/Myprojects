package com.saunak.director.ai

import java.security.MessageDigest

/**
 * Stable, standalone description of the camera hardware and Director-visible capabilities.
 *
 * The full payload is synchronized to Sierra/Gemini separately from image planning. Normal
 * photographic requests reference the stored Gemini interaction instead of resending this payload.
 */
data class DeviceFoundationProfile(
    val payload: String,
    val digest: String,
    val generatedAtMs: Long = System.currentTimeMillis()
) {
    init {
        require(payload.isNotBlank())
        require(digest.length == 64)
    }

    companion object {
        fun create(payload: String): DeviceFoundationProfile {
            val normalized = payload.trim().replace("\r\n", "\n")
            val digest = MessageDigest.getInstance("SHA-256")
                .digest(normalized.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
            return DeviceFoundationProfile(normalized, digest)
        }
    }
}
