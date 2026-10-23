def is_webp(raw: bytes) -> bool:
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"


def test_arbitrary_riff_is_not_webp():
    assert not is_webp(b"RIFF1234WAVEpayload")


def test_webp_signature_is_accepted():
    assert is_webp(b"RIFF1234WEBPpayload")
