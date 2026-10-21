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
    const val LOOK_NATURAL = 30
    const val LOOK_WARM = 31
    const val LOOK_COOL = 32
    const val LOOK_CINEMATIC_WARM = 33
    const val LOOK_CINEMATIC_COOL = 34
    const val LOOK_MONOCHROME = 35
    const val LOOK_SOFT_PORTRAIT = 36
    const val LOOK_LANDSCAPE_VIVID = 37
    val CAMERA_IDS = setOf(FOCUS, ZOOM, EXPOSURE_EV, TORCH, FLASH, EXTENSION)
    val LOOK_IDS = setOf(LOOK_NATURAL, LOOK_WARM, LOOK_COOL, LOOK_CINEMATIC_WARM, LOOK_CINEMATIC_COOL, LOOK_MONOCHROME, LOOK_SOFT_PORTRAIT, LOOK_LANDSCAPE_VIVID)
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
        add("30 natural(v:0..1 optional strength)")
        add("31 warm(v:0..1 optional strength)")
        add("32 cool(v:0..1 optional strength)")
        add("33 cinematic_warm(v:0..1 optional strength)")
        add("34 cinematic_cool(v:0..1 optional strength)")
        add("35 monochrome(v:0..1 optional strength)")
        add("36 soft_portrait(v:0..1 optional strength)")
        add("37 landscape_vivid(v:0..1 optional strength)")
    }.joinToString("; ")

    private fun fmt(v: Float): String = "%.2f".format(java.util.Locale.US, v)
}
