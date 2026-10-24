package com.saunak.director.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

private val AccountInk = Color(0xFF090A0C)
private val AccountGraphite = Color(0xFF111318)
private val AccountGold = Color(0xFFF2B84B)
private val AccountIvory = Color(0xFFF5F6F8)
private val AccountSilver = Color(0xFFA4ABB5)
private val AccountError = Color(0xFFFF5D5D)

@Composable
fun SierraVisionRoot(vm: DirectorViewModel = viewModel()) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = AccountInk,
            surface = AccountGraphite,
            primary = AccountGold
        )
    ) {
        Box(Modifier.fillMaxSize()) {
            SierraVisionApp(vm)
            if (ui.serverConfigured) {
                SierraAccountBadge(
                    signedIn = ui.signedIn,
                    credits = ui.creditsAvailable,
                    onClick = vm::openAccount,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .statusBarsPadding()
                        .padding(top = 64.dp, end = 16.dp)
                )
            }
            if (ui.showAccount) SierraAccountSheet(vm)
        }
    }
}

@Composable
private fun SierraAccountBadge(
    signedIn: Boolean,
    credits: Int?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        onClick = onClick,
        color = AccountGraphite.copy(alpha = .90f),
        contentColor = AccountIvory,
        shape = RoundedCornerShape(18.dp),
        modifier = modifier
    ) {
        Row(
            Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(Modifier.background(if (signedIn) AccountGold else AccountSilver, CircleShape).padding(3.dp))
            Text(
                if (signedIn) "${credits ?: "—"} ◉" else "Sign in",
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun SierraAccountSheet(vm: DirectorViewModel) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    BackHandler { vm.closeAccount() }

    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = .70f))
            .clickable { vm.closeAccount() }
            .padding(18.dp),
        contentAlignment = Alignment.Center
    ) {
        Surface(
            color = AccountGraphite,
            shape = RoundedCornerShape(28.dp),
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 520.dp)
                .navigationBarsPadding()
                .clickable(enabled = false) {}
        ) {
            if (ui.signedIn) SierraSignedInAccount(vm) else SierraAuthentication(vm)
        }
    }
}

@Composable
private fun SierraSignedInAccount(vm: DirectorViewModel) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("SIERRA ACCOUNT", color = AccountIvory, letterSpacing = 2.sp, fontWeight = FontWeight.Bold)
                Text(ui.userDisplayName ?: "Sierra photographer", color = AccountSilver, fontSize = 12.sp)
            }
            Text("✕", color = AccountIvory, modifier = Modifier.clickable { vm.closeAccount() }.padding(8.dp))
        }
        Surface(color = Color.White.copy(alpha = .045f), shape = RoundedCornerShape(20.dp)) {
            Column(Modifier.fillMaxWidth().padding(18.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text("${ui.creditsAvailable ?: 0}", color = AccountGold, fontSize = 36.sp, fontWeight = FontWeight.SemiBold)
                Text("Sierra AI photos available", color = AccountIvory, fontWeight = FontWeight.Medium)
                Text(
                    if (ui.creditsReserved > 0) "${ui.creditsReserved} credit reserved for an active shot" else "1 credit = 1 completed Sierra AI photograph",
                    color = AccountSilver,
                    fontSize = 11.sp
                )
            }
        }
        ui.userEmail?.let { Text(it, color = AccountSilver, fontSize = 12.sp) }
        if (ui.accountError != null) Text(ui.accountError ?: "", color = AccountError, fontSize = 12.sp)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TextButton(onClick = vm::refreshAccount, enabled = !ui.accountBusy) {
                Text("Refresh", color = AccountGold)
            }
            TextButton(onClick = vm::signOut, enabled = !ui.accountBusy) {
                Text("Sign out", color = AccountSilver)
            }
        }
        Text(
            "Sierra β remains available without credits or network. Private Photography never sends an analysis frame to the server.",
            color = AccountSilver,
            fontSize = 11.sp,
            lineHeight = 16.sp
        )
    }
}

@Composable
private fun SierraAuthentication(vm: DirectorViewModel) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    var signUp by remember { mutableStateOf(false) }
    var name by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var phone by remember { mutableStateOf("") }
    var dob by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(13.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("SIERRA VISION", color = AccountIvory, letterSpacing = 3.sp, fontWeight = FontWeight.Bold)
                Text(if (signUp) "Create your account" else "Sign in for Sierra AI", color = AccountSilver, fontSize = 12.sp)
            }
            Text("✕", color = AccountIvory, modifier = Modifier.clickable { vm.closeAccount() }.padding(8.dp))
        }

        if (signUp) {
            SierraField(name, { name = it }, "Name", KeyboardType.Text)
            SierraField(phone, { phone = it }, "Phone with country code, e.g. +91…", KeyboardType.Phone)
            SierraField(dob, { dob = it }, "Date of birth · YYYY-MM-DD", KeyboardType.Number)
        }
        SierraField(email, { email = it }, "Email", KeyboardType.Email)
        SierraField(password, { password = it }, "Password", KeyboardType.Password, password = true)

        if (ui.accountError != null) Text(ui.accountError ?: "", color = AccountError, fontSize = 12.sp)
        Button(
            onClick = {
                if (signUp) vm.signUp(name, email, phone, dob, password)
                else vm.signIn(email, password)
            },
            enabled = !ui.accountBusy,
            colors = ButtonDefaults.buttonColors(containerColor = AccountGold, contentColor = Color.Black),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (ui.accountBusy) CircularProgressIndicator(color = Color.Black, strokeWidth = 2.dp, modifier = Modifier.height(20.dp))
            else Text(if (signUp) "Create account · 5 trial credits" else "Sign in", fontWeight = FontWeight.SemiBold)
        }
        TextButton(onClick = { signUp = !signUp }, enabled = !ui.accountBusy) {
            Text(if (signUp) "Already have an account? Sign in" else "New to Sierra? Create account", color = AccountGold)
        }
        Spacer(Modifier.height(2.dp))
        Text(
            "You can close this and keep photographing with Sierra β. Sierra AI requires an account because credits and cloud jobs are controlled by your Sierra server.",
            color = AccountSilver,
            textAlign = TextAlign.Center,
            fontSize = 11.sp,
            lineHeight = 16.sp,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Composable
private fun SierraField(
    value: String,
    onValue: (String) -> Unit,
    label: String,
    keyboardType: KeyboardType,
    password: Boolean = false
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValue,
        label = { Text(label) },
        singleLine = true,
        visualTransformation = if (password) PasswordVisualTransformation() else androidx.compose.ui.text.input.VisualTransformation.None,
        keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
        modifier = Modifier.fillMaxWidth()
    )
}
