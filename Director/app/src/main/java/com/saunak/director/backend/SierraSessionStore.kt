package com.saunak.director.backend

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Stores only Sierra account session credentials, never VLM/provider credentials. */
class SierraSessionStore(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    data class Session(
        val accessToken: String,
        val refreshToken: String,
        val userId: String,
        val displayName: String,
        val creditsAvailable: Int
    )

    fun read(): Session? {
        val access = decrypt(KEY_ACCESS) ?: return null
        val refresh = decrypt(KEY_REFRESH) ?: return null
        val userId = prefs.getString(KEY_USER_ID, null) ?: return null
        return Session(
            accessToken = access,
            refreshToken = refresh,
            userId = userId,
            displayName = prefs.getString(KEY_DISPLAY_NAME, "").orEmpty(),
            creditsAvailable = prefs.getInt(KEY_CREDITS, 0)
        )
    }

    fun save(session: Session) {
        encrypt(KEY_ACCESS, session.accessToken)
        encrypt(KEY_REFRESH, session.refreshToken)
        prefs.edit()
            .putString(KEY_USER_ID, session.userId)
            .putString(KEY_DISPLAY_NAME, session.displayName)
            .putInt(KEY_CREDITS, session.creditsAvailable)
            .apply()
    }

    fun updateCredits(value: Int) {
        prefs.edit().putInt(KEY_CREDITS, value.coerceAtLeast(0)).apply()
    }

    fun clear() {
        prefs.edit().clear().apply()
    }

    private fun encrypt(name: String, value: String) {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val encrypted = cipher.doFinal(value.toByteArray(Charsets.UTF_8))
        prefs.edit()
            .putString("${name}_ct", Base64.encodeToString(encrypted, Base64.NO_WRAP))
            .putString("${name}_iv", Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .apply()
    }

    private fun decrypt(name: String): String? {
        val encrypted = prefs.getString("${name}_ct", null) ?: return null
        val iv = prefs.getString("${name}_iv", null) ?: return null
        return runCatching {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                key(),
                GCMParameterSpec(128, Base64.decode(iv, Base64.NO_WRAP))
            )
            String(cipher.doFinal(Base64.decode(encrypted, Base64.NO_WRAP)), Charsets.UTF_8)
        }.getOrNull()
    }

    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(ALIAS, null) as? SecretKey)?.let { return it }
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
        private const val PREFS = "sierra_account_session_v1"
        private const val KEY_ACCESS = "access"
        private const val KEY_REFRESH = "refresh"
        private const val KEY_USER_ID = "user_id"
        private const val KEY_DISPLAY_NAME = "display_name"
        private const val KEY_CREDITS = "credits"
        private const val ALIAS = "sierra_account_session_key_v1"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}
