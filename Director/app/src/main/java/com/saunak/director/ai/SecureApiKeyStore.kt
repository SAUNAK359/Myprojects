package com.saunak.director.ai

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Stores provider API keys encrypted with Android Keystore. */
class SecureApiKeyStore(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun hasKey(providerId: String = "nvidia"): Boolean = !prefs.getString(cipherName(providerId), null).isNullOrBlank()

    fun save(apiKey: String, providerId: String = "nvidia") {
        val clean = apiKey.trim()
        require(clean.length >= 12) { "API key looks incomplete" }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val encrypted = cipher.doFinal(clean.toByteArray(Charsets.UTF_8))
        prefs.edit()
            .putString(cipherName(providerId), Base64.encodeToString(encrypted, Base64.NO_WRAP))
            .putString(ivName(providerId), Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .apply()
    }

    fun read(providerId: String = "nvidia"): String? {
        val encrypted = prefs.getString(cipherName(providerId), null) ?: return null
        val iv = prefs.getString(ivName(providerId), null) ?: return null
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

    fun clear(providerId: String = "nvidia") {
        prefs.edit().remove(cipherName(providerId)).remove(ivName(providerId)).apply()
    }

    private fun cipherName(providerId: String) = "provider_${safe(providerId)}_ciphertext_v3"
    private fun ivName(providerId: String) = "provider_${safe(providerId)}_iv_v3"
    private fun safe(value: String) = value.lowercase().replace(Regex("[^a-z0-9_-]"), "_")

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
        private const val PREFS = "director_secure_settings"
        private const val ALIAS = "director_provider_keys_v3"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}
