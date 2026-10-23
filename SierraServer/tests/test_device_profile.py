import pytest
from pydantic import ValidationError

from app.schemas import DeviceProfileInput, SignupRequest


def sample_device() -> dict:
    return {
        "install_id": "test-install-00000001",
        "manufacturer": "TestMaker",
        "model": "TestPhone",
        "device_name": "test-device",
        "android_release": "16",
        "sdk_int": 36,
        "app_version": "1.0.0",
        "app_build": 1,
        "locale": "en",
        "timezone": "UTC",
        "screen_width_px": 1080,
        "screen_height_px": 2400,
        "density_dpi": 420,
        "is_emulator": False,
        "sierra_beta_version": "1",
        "camera_capabilities": {"raw": "false", "zoom_max": "8.0"},
    }


def test_device_profile_contains_only_operational_metadata():
    profile = DeviceProfileInput.model_validate(sample_device())
    serialized = profile.model_dump()
    assert profile.camera_capabilities == {"raw": "false", "zoom_max": "8.0"}
    for forbidden in ("imei", "serial", "advertising_id", "precise_location", "contacts"):
        assert forbidden not in serialized


def test_signup_requires_device_profile():
    payload = {
        "display_name": "Test User",
        "email": "user@example.invalid",
        "phone": "+0000000",
        "dob": "2000-01-01",
        "password": "test-password-long-enough",
    }
    with pytest.raises(ValidationError):
        SignupRequest.model_validate(payload)


def test_device_profile_rejects_old_android_sdk():
    payload = sample_device()
    payload["sdk_int"] = 21
    with pytest.raises(ValidationError):
        DeviceProfileInput.model_validate(payload)
