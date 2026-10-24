import pytest
from pydantic import ValidationError

from app.provider_client import parse_plan
from app.schemas import SierraPlan


def test_valid_sierra_plan():
    plan = SierraPlan.model_validate({
        "scene": "portrait",
        "objective": "protect highlights and isolate subject",
        "branch": "portrait",
        "reference_score": 0.82,
        "reference_accept": False,
        "calls": [{"id": 42, "v": 0.44, "x": None, "y": None, "s": None}],
        "recapture": False,
        "aspect_ratio": "4:5",
        "crop_center_x": 0.48,
        "crop_center_y": 0.46,
        "planner": "sierra_ai_server",
    })
    assert plan.aspect_ratio == "4:5"
    assert len(plan.calls) == 1


def test_invalid_ratio_rejected():
    with pytest.raises(ValidationError):
        SierraPlan.model_validate({
            "scene": "x", "objective": "y", "branch": "general", "reference_score": 0.5,
            "reference_accept": False, "calls": [], "recapture": False, "aspect_ratio": "5:7",
        })


def test_parser_skips_reasoning_object_before_plan():
    text = 'thinking {"step":1} final {"scene":"landscape","objective":"clean horizon","branch":"landscape","reference_score":0.7,"reference_accept":false,"calls":[],"recapture":false,"aspect_ratio":"16:9","crop_center_x":0.5,"crop_center_y":0.5,"planner":"sierra_ai_server"}'
    assert parse_plan(text).scene == "landscape"
