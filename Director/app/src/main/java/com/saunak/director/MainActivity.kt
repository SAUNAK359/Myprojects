package com.saunak.director

import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.saunak.director.ui.DirectorViewModel
import com.saunak.director.ui.SierraVisionApp
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val directorViewModel: DirectorViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // A compact native execution HUD guarantees that requested -> resolved -> actual camera
        // outcomes are visible even before the richer Sierra timeline UI is expanded.
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                directorViewModel.ui
                    .map { state ->
                        state.executionReceipts.takeIf { it.isNotEmpty() }
                            ?.joinToString("\n") { it.compactDescription() }
                    }
                    .distinctUntilChanged()
                    .collect { summary ->
                        if (!summary.isNullOrBlank()) {
                            Toast.makeText(this@MainActivity, summary, Toast.LENGTH_LONG).show()
                        }
                    }
            }
        }

        setContent { SierraVisionApp(directorViewModel) }
    }
}
