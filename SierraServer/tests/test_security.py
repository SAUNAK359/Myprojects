from app.security import hash_password, hash_token, verify_password


def test_password_is_hashed_and_verifies():
    raw = "correct horse battery staple 123"
    hashed = hash_password(raw)
    assert raw not in hashed
    assert verify_password(raw, hashed)
    assert not verify_password("wrong", hashed)


def test_refresh_hash_is_deterministic_and_not_plaintext():
    raw = "refresh-token-example"
    assert hash_token(raw) == hash_token(raw)
    assert raw not in hash_token(raw)
