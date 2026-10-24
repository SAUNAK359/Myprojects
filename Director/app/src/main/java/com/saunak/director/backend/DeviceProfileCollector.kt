package com.saunak.director.backend

import android.content.Context
import android.os.Build
import com.saunak.director.BuildConfig
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import kotlinx.serialization.Serializable

@Serializable
data class SierraDeviceProfile(
    val install_id: String,
    val manufacturer: String,
    val model: String,
    val device_name: String? = null,
    val android_release: String,
    val sdk_int: Int,
    val app_version: String,
    val app_build: Int,
    val locale: String,
    val timezone: String,
    val screen_width_px: Int? = null,
    val screen_height_px: Int? = null,
    val density_dpi: Int? = null,
    val is_emulator: Boolean = false,
    val sierra_beta_version: String? = "1",
    val camera_capabilities: Map<String, String>? = null
)

class DeviceProfileCollector(private val context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun collect(cameraCapabilities: Map<String, String>? = null): SierraDeviceProfile {
        val metrics = context.resources.displayMetrics
        return SierraDeviceProfile(
            install_id = installId(),
            manufacturer = Build.MANUFACTURER.orEmpty().ifBlank { "unknown" },
            model = Build.MODEL.orEmpty().ifBlank { "unknown" },
            device_name = Build.DEVICE?.takeIf { it.isNotBlank() },
            android_release = Build.VERSION.RELEASE.orEmpty().ifBlank { "unknown" },
            sdk_int = Build.VERSION.SDK_INT,
            app_version = BuildConfig.VERSION_NAME,
            app_build = BuildConfig.VERSION_CODE,
            locale = Locale.getDefault().toLanguageTag().ifBlank { "und" },
            timezone = TimeZone.getDefault().id.ifBlank { "UTC" },
            screen_width_px = metrics.widthPixels.takeIf { it > 0 },
            screen_height_px = metrics.heightPixels.takeIf { it > 0 },
            density_dpi = metrics.densityDpi.takeIf { it > 0 },
            is_emulator = isProbablyEmulator(),
            camera_capabilities = cameraCapabilities
        )
    }

    private fun installId(): String {
        prefs.getString(KEY_INSTALL_ID, null)?.let { return it }
        val value = UUID.randomUUID().toString()
        prefs.edit().putString(KEY_INSTALL_ID, value).apply()
        return value
    }

    private fun isProbablyEmulator(): Boolean {
        val fingerprint = Build.FINGERPRINT.orEmpty().lowercase(Locale.US)
        val model = Build.MODEL.orEmpty().lowercase(Locale.US)
        val product = Build.PRODUCT.orEmpty().lowercase(Locale.US)
        return fingerprint.contains("generic") ||
            fingerprint.contains("emulator") ||
            model.contains("sdk_gphone") ||
            model.contains("emulator") ||
            product.contains("sdk")
    }

    companion object {
        private const val PREFS = "sierra_install_identity_v1"
        private const val KEY_INSTALL_ID = "install_id"
    }
}
