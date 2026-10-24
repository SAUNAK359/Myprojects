from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


def _fernet() -> Fernet:
    value = settings.provider_master_key.get_secret_value().strip()
    if not value:
        raise RuntimeError("SIERRA_PROVIDER_MASTER_KEY is not configured")
    try:
        return Fernet(value.encode())
    except Exception as exc:
        raise RuntimeError("SIERRA_PROVIDER_MASTER_KEY must be a valid Fernet key") from exc


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise RuntimeError("provider secret cannot be decrypted with current master key") from exc
