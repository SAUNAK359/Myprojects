package com.saunak.director

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.saunak.director.ai.GeminiApiKeyStore

/**
 * Beta credential gate. The user types the Gemini key on the device; the plaintext is never placed
 * in source code, BuildConfig or preferences. GeminiApiKeyStore encrypts it with Android Keystore.
 */
class GeminiSetupActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val store = GeminiApiKeyStore(this)
        setContent {
            GeminiSetupScreen(
                configuredAtStart = store.isConfigured(),
                onSave = { candidate -> runCatching { store.save(candidate) } },
                onClear = { store.clear() },
                onContinue = { openCamera() }
            )
        }
    }

    private fun openCamera() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}

@Composable
private fun GeminiSetupScreen(
    configuredAtStart: Boolean,
    onSave: (String) -> Result<Unit>,
    onClear: () -> Unit,
    onContinue: () -> Unit
) {
    val ink = Color(0xFF090A0C)
    val graphite = Color(0xFF111318)
    val gold = Color(0xFFF2B84B)
    val ivory = Color(0xFFF5F6F8)
    val silver = Color(0xFFA4ABB5)
    val success = Color(0xFF6FD99C)
    val error = Color(0xFFFF5D5D)

    var key by remember { mutableStateOf("") }
    var configured by remember { mutableStateOf(configuredAtStart) }
    var message by remember { mutableStateOf(if (configuredAtStart) "Gemini 3.8 Flash is configured on this phone." else "Add your Gemini API key for Sierra AI.") }
    var isError by remember { mutableStateOf(false) }

    MaterialTheme(colorScheme = darkColorScheme(background = ink, surface = graphite, primary = gold)) {
        Box(Modifier.fillMaxSize().background(ink), contentAlignment = Alignment.Center) {
            Column(
                Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().verticalScroll(rememberScrollState()).padding(horizontal = 26.dp, vertical = 34.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Surface(color = Color.Transparent, shape = CircleShape, border = androidx.compose.foundation.BorderStroke(2.dp, gold), modifier = Modifier.padding(8.dp)) {
                    Box(Modifier.padding(horizontal = 25.dp, vertical = 19.dp), contentAlignment = Alignment.Center) {
                        Text("S", color = gold, fontSize = 31.sp, fontWeight = FontWeight.Light)
                    }
                }
                Spacer(Modifier.height(16.dp))
                Text("SIERRA AI SETUP", color = ivory, fontSize = 22.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 3.sp)
                Text("Beta · direct Gemini supervision", color = silver, fontSize = 13.sp, modifier = Modifier.padding(top = 6.dp, bottom = 24.dp))

                Surface(color = graphite, shape = RoundedCornerShape(22.dp), modifier = Modifier.fillMaxWidth().widthIn(max = 540.dp)) {
                    Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                        Text(
                            "The key is encrypted with Android Keystore and remains on this device. Sierra sends only reduced analysis frames to Gemini when AI supervision is enabled. Camera control is still validated locally by Sierra.",
                            color = silver,
                            fontSize = 12.sp,
                            lineHeight = 18.sp
                        )
                        OutlinedTextField(
                            value = key,
                            onValueChange = { key = it; isError = false },
                            label = { Text(if (configured) "Replace Gemini API key" else "Gemini API key") },
                            placeholder = { Text("Paste key from Google AI Studio") },
                            singleLine = true,
                            visualTransformation = PasswordVisualTransformation(),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Text(message, color = if (isError) error else if (configured) success else silver, fontSize = 11.sp)
                        Button(
                            onClick = {
                                val result = onSave(key)
                                if (result.isSuccess) {
                                    configured = true
                                    key = ""
                                    isError = false
                                    message = "Gemini key saved securely on this phone."
                                } else {
                                    isError = true
                                    message = result.exceptionOrNull()?.message ?: "Could not save Gemini key"
                                }
                            },
                            enabled = key.isNotBlank(),
                            colors = ButtonDefaults.buttonColors(containerColor = gold, contentColor = Color.Black),
                            shape = RoundedCornerShape(16.dp),
                            modifier = Modifier.fillMaxWidth().height(50.dp)
                        ) { Text(if (configured) "Replace key" else "Save key", fontWeight = FontWeight.Bold) }

                        if (configured) {
                            OutlinedButton(
                                onClick = {
                                    onClear()
                                    configured = false
                                    isError = false
                                    message = "Gemini key removed. Sierra β will remain available locally."
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) { Text("Remove key", color = silver) }
                        }
                    }
                }

                Spacer(Modifier.height(18.dp))
                Button(
                    onClick = onContinue,
                    colors = ButtonDefaults.buttonColors(containerColor = ivory, contentColor = Color.Black),
                    shape = RoundedCornerShape(17.dp),
                    modifier = Modifier.fillMaxWidth().widthIn(max = 540.dp).height(52.dp)
                ) { Text(if (configured) "Open Sierra AI Camera" else "Open Sierra β Camera", fontWeight = FontWeight.Bold) }
                TextButton(onClick = onContinue) { Text("Continue without Gemini", color = silver) }
                Text(
                    "For production, provider authentication should move behind Firebase AI Logic or your backend; this direct-key mode is intentionally a personal beta path.",
                    color = silver.copy(alpha = .75f),
                    fontSize = 10.sp,
                    lineHeight = 15.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.widthIn(max = 520.dp)
                )
            }
        }
    }
}
