package com.saunak.director.ai

import org.json.JSONArray
import org.json.JSONObject

internal object SierraGeminiSchemas {
    private fun number() = JSONObject().put("type", "number")
    private fun integer() = JSONObject().put("type", "integer")
    private fun string() = JSONObject().put("type", "string")
    private fun bool() = JSONObject().put("type", "boolean")
    private fun array(item: JSONObject, max: Int? = null) = JSONObject().put("type", "array").put("items", item).apply { max?.let { put("maxItems", it) } }
    private fun obj(required: List<String>, properties: Map<String, JSONObject>) = JSONObject()
        .put("type", "object")
        .put("required", JSONArray(required))
        .put("properties", JSONObject(properties))
        .put("additionalProperties", false)

    fun plan(directorContract: DirectorSchemaCompiler.Contract): JSONObject {
        val nullableNumber = JSONObject().put("type", JSONArray(listOf("number", "null")))
        val nullableString = JSONObject().put("type", JSONArray(listOf("string", "null")))
        val call = obj(listOf("id","v","x","y","s"), mapOf("id" to integer(), "v" to nullableNumber, "x" to nullableNumber, "y" to nullableNumber, "s" to nullableString))
        val blend = obj(listOf("theme_id","weight"), mapOf("theme_id" to string(), "weight" to number()))
        val directionKeys = listOf("story","mood","primary_subject","narrative_role","selected_theme_id","theme_strength","theme_confidence","theme_blend","atmosphere_priority","skin_naturalness","highlight_protection","shadow_depth","midtone_contrast","local_contrast","color_intensity","warmth_bias","subject_relight_ev","background_suppression_ev","grain","vignette","abstain","reason_codes")
        val direction = obj(directionKeys, mapOf(
            "story" to string(), "mood" to string(), "primary_subject" to string(), "narrative_role" to string(), "selected_theme_id" to string(),
            "theme_strength" to number(), "theme_confidence" to number(), "theme_blend" to array(blend, 3), "atmosphere_priority" to number(),
            "skin_naturalness" to number(), "highlight_protection" to number(), "shadow_depth" to number(), "midtone_contrast" to number(),
            "local_contrast" to number(), "color_intensity" to number(), "warmth_bias" to number(), "subject_relight_ev" to number(),
            "background_suppression_ev" to number(), "grain" to number(), "vignette" to number(), "abstain" to bool(), "reason_codes" to array(string(), 8)
        ))
        val keys = listOf("scene","objective","branch","reference_score","reference_accept","calls","recapture","aspect_ratio","crop_center_x","crop_center_y","photographic_direction","director_spec","planner")
        return obj(keys, mapOf(
            "scene" to string(), "objective" to string(), "branch" to string(), "reference_score" to number(), "reference_accept" to bool(),
            "calls" to array(call, 8), "recapture" to bool(), "aspect_ratio" to string(), "crop_center_x" to number(), "crop_center_y" to number(),
            "photographic_direction" to direction, "director_spec" to directorContract.jsonSchema, "planner" to string()
        ))
    }

    fun evaluation(): JSONObject {
        fun section(keys: List<String>): JSONObject = obj(keys, keys.associateWith { number() })
        val tonal = section(listOf("exposure_ev","black_point","toe_strength","midtone_contrast","shoulder_strength","dynamic_range_compression"))
        val color = section(listOf("temperature","tint","saturation","vibrance","shadow_hue_shift_deg","shadow_color_strength","highlight_hue_shift_deg","highlight_color_strength","skin_protection"))
        val local = section(listOf("subject_ev","background_ev","face_ev","rim_emphasis","dodge_burn_strength"))
        val hierarchy = section(listOf("subject_separation","background_presence","luminous_source_priority","text_preservation"))
        val materials = section(listOf("skin_texture","hair_detail","fabric_detail","foliage_detail","metal_specular","glass_specular"))
        val highlights = section(listOf("rolloff","compression","bloom","halation","practical_light_preservation"))
        val shadows = section(listOf("depth","recovery","black_anchor","color_strength"))
        val texture = section(listOf("denoise","chroma_denoise","detail","microcontrast","sharpening","deblur"))
        val optical = section(listOf("diffusion","vignette","edge_softness","lens_falloff"))
        val film = obj(listOf("curve_strength","grain_amount","grain_size","color_response","character"), mapOf("curve_strength" to number(),"grain_amount" to number(),"grain_size" to number(),"color_response" to number(),"character" to string()))
        val atmosphere = section(listOf("dehaze","glow","haze","local_contrast"))
        val spatial = obj(listOf("use_subject_mask","use_depth","use_face_mask","use_skin_mask","use_sky_mask","feather"), mapOf("use_subject_mask" to bool(),"use_depth" to bool(),"use_face_mask" to bool(),"use_skin_mask" to bool(),"use_sky_mask" to bool(),"feather" to number()))
        val finishing = obj(listOf("output_contrast","output_saturation","output_sharpen","preserve_blacks"), mapOf("output_contrast" to number(),"output_saturation" to number(),"output_sharpen" to number(),"preserve_blacks" to bool()))
        val execution = obj(listOf("working_long_edge","precision","subject_mask_method","depth_method","operations","device_rationale"), mapOf("working_long_edge" to integer(),"precision" to string(),"subject_mask_method" to string(),"depth_method" to string(),"operations" to array(string(),32),"device_rationale" to string()))
        val renderKeys = listOf("schema_version","intent_summary","tonal_response","color_science","local_light","subject_background_hierarchy","material_rendering","highlight_behavior","shadow_behavior","texture_model","optical_character","film_response","atmosphere","spatial_masks","finishing","execution","reason_codes")
        val render = obj(renderKeys, mapOf(
            "schema_version" to integer(),"intent_summary" to string(),"tonal_response" to tonal,"color_science" to color,"local_light" to local,
            "subject_background_hierarchy" to hierarchy,"material_rendering" to materials,"highlight_behavior" to highlights,"shadow_behavior" to shadows,
            "texture_model" to texture,"optical_character" to optical,"film_response" to film,"atmosphere" to atmosphere,"spatial_masks" to spatial,
            "finishing" to finishing,"execution" to execution,"reason_codes" to array(string(),8)
        ))
        val adjustmentKeys = listOf("theme_strength_delta","midtone_contrast_delta","local_contrast_delta","shadow_depth_delta","highlight_protection_delta","subject_relight_delta_ev","background_suppression_delta_ev","warmth_delta","color_intensity_delta")
        val adjustment = section(adjustmentKeys)
        val keys = listOf("score","technical","composition","subject","aesthetic","intent_match","naturalness","accept","issues","delta","render_adjustment","render_spec")
        return obj(keys, mapOf(
            "score" to number(),"technical" to number(),"composition" to number(),"subject" to number(),"aesthetic" to number(),
            "intent_match" to number(),"naturalness" to number(),"accept" to bool(),"issues" to array(string(),8),
            "delta" to array(JSONObject().put("type","object"),0),"render_adjustment" to adjustment,"render_spec" to render
        ))
    }
}
