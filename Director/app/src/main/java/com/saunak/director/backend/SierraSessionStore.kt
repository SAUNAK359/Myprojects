package com.saunak.director.backend

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

@Serializable
data class SierraStoredSession(
    val accessToken: String,
    val refreshToken: String,
    val accessExpiresAtMs: Long,
    val user: SierraUserView
)

class SierraSessionStore(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val json = Json { ignoreUnknownKeys = true }

    fun save(pair: SierraTokenPair) {
        val value = SierraStoredSession(
            accessToken = pair.accessToken,
            refreshToken = pair.refreshToken,
            accessExpiresAtMs = System.currentTimeMillis() + pair.accessExpiresInSeconds * 1_000L,
            user = pair.user
        )
        encrypt(json.encodeToString(value))
    }

    fun read(): SierraStoredSession? {
        val encoded = decrypt() ?: return null
        return runCatching { json.decodeFromString<SierraStoredSession>(encoded) }.getOrNull()
    }

    fun clear() {
        prefs.edit().remove(CIPHERTEXT).remove(IV).apply()
    }

    private fun encrypt(value: String) {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val encrypted = cipher.doFinal(value.toByteArray(Charsets.UTF_8))
        prefs.edit()
            .putString(CIPHERTEXT, Base64.encodeToString(encrypted, Base64.NO_WRAP))
            .putString(IV, Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .apply()
    }

    private fun decrypt(): String? {
        val encrypted = prefs.getString(CIPHERTEXT, null) ?: return null
        val iv = prefs.getString(IV, null) ?: return null
        return runCatching {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                getOrCreateKey(),
                GCMParameterSpec(128, Base64.decode(iv, Base64.NO_WRAP))
            )
            String(cipher.doFinal(Base64.decode(encrypted, Base64.NO_WRAP)), Charsets.UTF_8)
        }.getOrNull()
    }

    private fun getOrCreateKey(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (keyStore.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(
                ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return generator.generateKey()
    }

    companion object {
        private const val PREFS = "sierra_secure_session_v1"
        private const val CIPHERTEXT = "session_ciphertext"
        private const val IV = "session_iv"
        private const val ALIAS = "sierra_account_session_v1"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}
