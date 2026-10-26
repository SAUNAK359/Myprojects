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

/**
 * Beta-only client-side Gemini credential vault.
 *
 * The API key is never compiled into the APK or committed to source. The stored preference contains
 * only AES-GCM ciphertext + IV; the non-exportable AES key lives in Android Keystore. This protects
 * the credential at rest, but a client-side credential is still not a production secret boundary on
 * a rooted/compromised device. Production Sierra should move provider auth behind Firebase/backend.
 */
class GeminiApiKeyStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun isConfigured(): Boolean = runCatching { read().isNotBlank() }.getOrDefault(false)

    fun read(): String {
        val encoded = prefs.getString(KEY_CIPHERTEXT, null) ?: return ""
        val pieces = encoded.split(':', limit = 2)
        if (pieces.size != 2) return ""
        val iv = Base64.decode(pieces[0], Base64.NO_WRAP)
        val ciphertext = Base64.decode(pieces[1], Base64.NO_WRAP)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), GCMParameterSpec(TAG_BITS, iv))
        return cipher.doFinal(ciphertext).toString(Charsets.UTF_8)
    }

    fun save(rawKey: String) {
        val key = normalize(rawKey)
        require(key.length in MIN_KEY_LENGTH..MAX_KEY_LENGTH) { "Gemini API key length is invalid" }
        require(key.none(Char::isWhitespace)) { "Gemini API key cannot contain spaces" }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val ciphertext = cipher.doFinal(key.toByteArray(Charsets.UTF_8))
        val payload = Base64.encodeToString(cipher.iv, Base64.NO_WRAP) + ":" +
            Base64.encodeToString(ciphertext, Base64.NO_WRAP)
        check(prefs.edit().putString(KEY_CIPHERTEXT, payload).commit()) { "Could not store Gemini API key" }
    }

    fun clear() {
        prefs.edit().remove(KEY_CIPHERTEXT).apply()
    }

    private fun getOrCreateKey(): SecretKey {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (store.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }

    companion object {
        private const val PREFS = "sierra_gemini_key_v1"
        private const val KEY_CIPHERTEXT = "gemini_key_ciphertext"
        private const val KEYSTORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "sierra_gemini_beta_aes_v1"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val TAG_BITS = 128
        private const val MIN_KEY_LENGTH = 16
        private const val MAX_KEY_LENGTH = 512

        fun normalize(value: String): String = value.trim()
    }
}
