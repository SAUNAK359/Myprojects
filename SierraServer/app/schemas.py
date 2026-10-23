import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

SUPPORTED_RATIOS = {"4:3", "3:2", "4:5", "1:1", "16:9", "9:16", "2.39:1"}


class ToolCall(BaseModel):
    id: int
    v: float | None = None
    x: float | None = None
    y: float | None = None
    s: str | None = None


class SierraPlan(BaseModel):
    scene: str = Field(min_length=1, max_length=64)
    objective: str = Field(min_length=1, max_length=180)
    branch: Literal["general", "portrait", "landscape", "low_light", "motion", "group", "food", "architecture", "macro", "selfie"]
    reference_score: float = Field(ge=0, le=1)
    reference_accept: bool = False
    calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    recapture: bool = True
    aspect_ratio: str = "4:3"
    crop_center_x: float = Field(0.5, ge=0, le=1)
    crop_center_y: float = Field(0.5, ge=0, le=1)
    planner: str = "sierra_ai_server"

    @field_validator("aspect_ratio")
    @classmethod
    def ratio_supported(cls, value: str) -> str:
        if value not in SUPPORTED_RATIOS:
            raise ValueError("unsupported aspect ratio")
        return value


class SignupRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    phone: str = Field(min_length=7, max_length=32)
    dob: date
    password: str = Field(min_length=12, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=32, max_length=300)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(min_length=32, max_length=300)


class UserView(BaseModel):
    id: uuid.UUID
    display_name: str
    email: EmailStr
    phone: str
    role: str
    credit_balance: int
    credit_reserved: int
    available_credits: int
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    access_expires_in_seconds: int
    user: UserView


class CreditView(BaseModel):
    balance: int
    reserved: int
    available: int


class JobView(BaseModel):
    id: uuid.UUID
    state: str
    plan: SierraPlan | None = None
    error_code: str | None = None
    error_message: str | None = None
    provider_name: str | None = None
    provider_model: str | None = None
    created_at: datetime
    ready_expires_at: datetime | None = None
    credits: CreditView


class CompleteJobRequest(BaseModel):
    capture_id: str = Field(min_length=8, max_length=120)


class ProviderUpsert(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    endpoint: str = Field(min_length=12, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(min_length=1, max_length=1000)
    enabled: bool = True
    priority: int = Field(100, ge=1, le=1000)
    timeout_ms: int = Field(5000, ge=1000, le=15000)
    max_tokens: int = Field(1100, ge=256, le=4096)
    supports_json_schema: bool = False

    @field_validator("endpoint")
    @classmethod
    def endpoint_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("provider endpoint must use HTTPS")
        return value.rstrip("/")


class ProviderView(BaseModel):
    id: uuid.UUID
    name: str
    endpoint: str
    model: str
    enabled: bool
    priority: int
    timeout_ms: int
    max_tokens: int
    supports_json_schema: bool
    success_count: int
    failure_count: int
    avg_latency_ms: int
    last_error: str | None = None
    has_secret: bool = True


class AdminCreditGrant(BaseModel):
    credits: int = Field(ge=1, le=10000)
    note: str = Field(default="admin grant", max_length=300)
