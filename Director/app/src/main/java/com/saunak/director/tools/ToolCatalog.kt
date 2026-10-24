package com.saunak.director.tools

import com.saunak.director.camera.CapabilityRegistry
import com.saunak.director.camera.DirectorExtension

object ToolIds {
    const val FOCUS = 1
    const val ZOOM = 2
    const val EXPOSURE_EV = 3
    const val TORCH = 4
    const val FLASH = 5
    const val EXTENSION = 6
    const val COMPOSITION = 7

    const val LOOK_NATURAL = 30
    const val LOOK_WARM = 31
    const val LOOK_COOL = 32
    const val LOOK_CINEMATIC_WARM = 33
    const val LOOK_CINEMATIC_COOL = 34
    const val LOOK_MONOCHROME = 35
    const val LOOK_SOFT_PORTRAIT = 36
    const val LOOK_LANDSCAPE_VIVID = 37
    const val LOOK_SOCIAL_POP = 38
    const val LOOK_FILM_TEAL_ORANGE = 39
    const val LOOK_GOLDEN_HOUR = 40
    const val LOOK_MOODY = 41
    const val LOOK_CLEAN_PORTRAIT = 42
    const val LOOK_NIGHT_NEON = 43

    val CAMERA_IDS = setOf(FOCUS, ZOOM, EXPOSURE_EV, TORCH, FLASH, EXTENSION)
    val LOOK_IDS = setOf(
        LOOK_NATURAL,
        LOOK_WARM,
        LOOK_COOL,
        LOOK_CINEMATIC_WARM,
        LOOK_CINEMATIC_COOL,
        LOOK_MONOCHROME,
        LOOK_SOFT_PORTRAIT,
        LOOK_LANDSCAPE_VIVID,
        LOOK_SOCIAL_POP,
        LOOK_FILM_TEAL_ORANGE,
        LOOK_GOLDEN_HOUR,
        LOOK_MOODY,
        LOOK_CLEAN_PORTRAIT,
        LOOK_NIGHT_NEON
    )
}

object ToolCatalog {
    fun contract(c: CapabilityRegistry, activeExtension: DirectorExtension = c.activeExtension): String = buildList {
        if (activeExtension == DirectorExtension.NONE) {
            if (ToolIds.FOCUS in c.allowedToolIds) add("1 focus(x:0..1,y:0..1) recapture")
            if (ToolIds.ZOOM in c.allowedToolIds) add("2 zoom(v:${fmt(c.minZoom)}..${fmt(c.maxZoom)}) recapture")
            if (ToolIds.EXPOSURE_EV in c.allowedToolIds) add("3 exposure_ev(v:${fmt(c.minExposureEv)}..${fmt(c.maxExposureEv)}) recapture")
            if (ToolIds.TORCH in c.allowedToolIds) add("4 torch(v:0|1) recapture")
            if (ToolIds.FLASH in c.allowedToolIds) add("5 still_flash(s:off|auto|on) recapture; exclusive with torch-on")
        }
        if (ToolIds.EXTENSION in c.allowedToolIds) {
            val modes = buildList {
                if (activeExtension != DirectorExtension.NONE) add("none")
                addAll(c.extensions.map { it.wireName })
            }.distinct()
            if (modes.isNotEmpty()) add("6 extension(s:${modes.joinToString("|")}) recapture; exclusive with focus/zoom/exposure/torch/flash")
        }

        // Semantic composition intent. Sierra AI may request one of these ratios and an optional
        // normalized crop center. Sierra β consumes this before hardware execution, validates it,
        // and applies the crop locally after the full-quality capture. It never reaches Camera2.
        add("7 composition(s:4:3|3:2|4:5|1:1|16:9|9:16|2.39:1,x:0..1 optional crop center,y:0..1 optional crop center) no recapture")

        add("30 natural(v:0..1 optional strength)")
        add("31 warm(v:0..1 optional strength)")
        add("32 cool(v:0..1 optional strength)")
        add("33 cinematic_warm(v:0..1 optional strength)")
        add("34 cinematic_cool(v:0..1 optional strength)")
        add("35 monochrome(v:0..1 optional strength)")
        add("36 soft_portrait(v:0..1 optional strength)")
        add("37 landscape_vivid(v:0..1 optional strength)")
        add("38 social_pop(v:0..1 optional strength)")
        add("39 film_teal_orange(v:0..1 optional strength)")
        add("40 golden_hour(v:0..1 optional strength)")
        add("41 moody(v:0..1 optional strength)")
        add("42 clean_portrait(v:0..1 optional strength)")
        add("43 night_neon(v:0..1 optional strength)")
    }.joinToString("; ")

    private fun fmt(v: Float): String = "%.2f".format(java.util.Locale.US, v)
}
