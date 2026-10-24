from app.provider_client import PHOTOGRAPHY_PROMPT, build_prompt


def test_prompt_never_equates_orientation_with_portrait():
    assert "NEVER infer portrait from frame/device orientation" in PHOTOGRAPHY_PROMPT
    assert "Tiny or distant people" in PHOTOGRAPHY_PROMPT
    assert "BOKEH" in PHOTOGRAPHY_PROMPT


def test_prompt_includes_measured_dynamic_range_context():
    quality = '{"highlight_p95":234,"top_mean_luma":204,"bottom_mean_luma":136,"top_bottom_delta":68}'
    prompt = build_prompt(
        camera_context_json='{"zoom":1.0}',
        device_capabilities_json='{"extensions":["hdr"]}',
        local_quality_json=quality,
        skill_context="landscape: protect sky",
    )
    assert quality in prompt
    assert "highlight_p95" in prompt
    assert "protect the sky first" in prompt
