import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import phonenumbers
from fastapi import HTTPException, status
from pwdlib import PasswordHash

from app.config import settings

password_hash = PasswordHash.recommended()
ALGORITHM = "HS256"


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_phone(value: str) -> str:
    try:
        parsed = phonenumbers.parse(value, None)
    except phonenumbers.NumberParseException as exc:
        raise ValueError("phone must include country code, e.g. +91...") from exc
    if not phonenumbers.is_valid_number(parsed):
        raise ValueError("invalid phone number")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def hash_password(value: str) -> str:
    return password_hash.hash(value)


def verify_password(value: str, hashed: str) -> bool:
    return password_hash.verify(value, hashed)


def create_access_token(user_id: uuid.UUID, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_minutes)).timestamp()),
        "jti": secrets.token_hex(12),
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret.get_secret_value(), algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise ValueError("wrong token type")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired access token") from exc


def new_refresh_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hash_token(raw)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def refresh_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_days)


def client_ip(headers: dict[str, str], fallback: str | None) -> str | None:
    forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or fallback
