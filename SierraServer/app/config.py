from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIERRA_", env_file=".env", extra="ignore")

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8080
    database_url: str = "postgresql+asyncpg://sierra:sierra_change_me@localhost:5432/sierra"
    jwt_secret: SecretStr = SecretStr("development-only-change-me-please-32-bytes")
    provider_master_key: SecretStr = SecretStr("")
    access_token_minutes: int = Field(15, ge=5, le=120)
    refresh_token_days: int = Field(30, ge=1, le=180)
    trial_credits: int = Field(5, ge=0, le=100)
    job_ready_minutes: int = Field(10, ge=2, le=60)
    max_upload_bytes: int = Field(4 * 1024 * 1024, ge=256 * 1024, le=20 * 1024 * 1024)
    max_image_dimension: int = Field(704, ge=384, le=1280)
    min_user_age: int = Field(0, ge=0, le=21)
    allowed_origins: str = ""
    public_base_url: str = ""
    admin_bootstrap_email: str = ""
    admin_bootstrap_password: SecretStr = SecretStr("")
    admin_bootstrap_phone: str = ""
    admin_bootstrap_dob: str = "1990-01-01"
    domain: str = ""
    runtime_dir: Path = Path("runtime")

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


def validate_runtime_config(value: Settings) -> None:
    jwt_value = value.jwt_secret.get_secret_value()
    if len(jwt_value) < 32:
        raise RuntimeError("SIERRA_JWT_SECRET must be at least 32 characters")

    if value.is_production:
        provider_key = value.provider_master_key.get_secret_value().strip()
        if "development-only" in jwt_value.lower() or "change_me" in jwt_value.lower():
            raise RuntimeError("production requires a generated SIERRA_JWT_SECRET")
        if not provider_key:
            raise RuntimeError("production requires SIERRA_PROVIDER_MASTER_KEY")
        if "sierra_change_me" in value.database_url:
            raise RuntimeError("production database password must be changed")
        if not value.public_base_url.startswith("https://"):
            raise RuntimeError("production requires an HTTPS SIERRA_PUBLIC_BASE_URL")


def prepare_runtime_directories(value: Settings) -> None:
    value.runtime_dir.mkdir(parents=True, exist_ok=True)
    (value.runtime_dir / "analysis").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
