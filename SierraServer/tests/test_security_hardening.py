import pytest

from app.network_security import validate_public_https_endpoint


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/v1/chat/completions",
        "https://localhost/v1/chat/completions",
        "https://127.0.0.1/v1/chat/completions",
        "https://[::1]/v1/chat/completions",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.8/v1/chat/completions",
        "https://192.168.1.20/v1/chat/completions",
        "https://user:pass@example.com/v1/chat/completions",
        "https://example.com:8443/v1/chat/completions",
    ],
)
def test_provider_endpoint_rejects_unsafe_targets(value: str):
    with pytest.raises(ValueError):
        validate_public_https_endpoint(value)
