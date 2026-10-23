import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.credits import grant
from app.db import get_db
from app.deps import admin_user
from app.models import AuditLog, Job, Provider, User
from app.provider_crypto import encrypt_secret
from app.schemas import AdminCreditGrant, ProviderUpsert, ProviderView

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="app/templates")


def provider_view(value: Provider) -> ProviderView:
    return ProviderView(
        id=value.id, name=value.name, endpoint=value.endpoint, model=value.model,
        enabled=value.enabled, priority=value.priority, timeout_ms=value.timeout_ms,
        max_tokens=value.max_tokens, supports_json_schema=value.supports_json_schema,
        success_count=value.success_count, failure_count=value.failure_count,
        avg_latency_ms=value.avg_latency_ms, last_error=value.last_error,
    )


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html", context={})


@router.get("/summary")
async def summary(_: User = Depends(admin_user), db: AsyncSession = Depends(get_db)) -> dict:
    users = await db.scalar(select(func.count()).select_from(User)) or 0
    active_jobs = await db.scalar(select(func.count()).select_from(Job).where(Job.state.in_(["queued", "processing", "ready"]))) or 0
    completed = await db.scalar(select(func.count()).select_from(Job).where(Job.state == "completed")) or 0
    providers = await db.scalar(select(func.count()).select_from(Provider).where(Provider.enabled.is_(True))) or 0
    credits = await db.scalar(select(func.coalesce(func.sum(User.credit_balance), 0))) or 0
    reserved = await db.scalar(select(func.coalesce(func.sum(User.credit_reserved), 0))) or 0
    return {"users": users, "active_jobs": active_jobs, "completed_jobs": completed, "enabled_providers": providers, "credits_outstanding": credits, "credits_reserved": reserved}


@router.get("/users")
async def users(_: User = Depends(admin_user), db: AsyncSession = Depends(get_db), limit: int = 100) -> list[dict]:
    rows = list((await db.scalars(select(User).order_by(User.created_at.desc()).limit(min(max(limit, 1), 500)))).all())
    return [{"id": str(u.id), "name": u.display_name, "email": u.email, "phone": u.phone, "role": u.role, "active": u.is_active, "balance": u.credit_balance, "reserved": u.credit_reserved, "available": u.available_credits, "created_at": u.created_at} for u in rows]


@router.get("/jobs")
async def jobs(_: User = Depends(admin_user), db: AsyncSession = Depends(get_db), limit: int = 100) -> list[dict]:
    rows = list((await db.scalars(select(Job).order_by(Job.created_at.desc()).limit(min(max(limit, 1), 500)))).all())
    return [{"id": str(j.id), "user_id": str(j.user_id), "state": j.state, "provider": j.provider_name, "model": j.provider_model, "error": j.error_message, "created_at": j.created_at, "completed_at": j.completed_at} for j in rows]


@router.get("/providers", response_model=list[ProviderView])
async def providers(_: User = Depends(admin_user), db: AsyncSession = Depends(get_db)) -> list[ProviderView]:
    values = list((await db.scalars(select(Provider).order_by(Provider.priority))).all())
    return [provider_view(value) for value in values]


@router.post("/providers", response_model=ProviderView, status_code=201)
async def add_provider(payload: ProviderUpsert, admin: User = Depends(admin_user), db: AsyncSession = Depends(get_db)) -> ProviderView:
    if await db.scalar(select(Provider.id).where(Provider.name == payload.name.strip())):
        raise HTTPException(status_code=409, detail="provider name already exists")
    provider = Provider(
        name=payload.name.strip(), endpoint=payload.endpoint, model=payload.model.strip(),
        secret_ciphertext=encrypt_secret(payload.api_key), enabled=payload.enabled, priority=payload.priority,
        timeout_ms=payload.timeout_ms, max_tokens=payload.max_tokens, supports_json_schema=payload.supports_json_schema,
    )
    db.add(provider)
    db.add(AuditLog(actor_user_id=admin.id, action="provider.create", target_type="provider", target_id=payload.name, details={"endpoint": payload.endpoint, "model": payload.model}))
    await db.commit()
    await db.refresh(provider)
    return provider_view(provider)


@router.patch("/providers/{provider_id}/enabled", response_model=ProviderView)
async def set_provider_enabled(provider_id: uuid.UUID, enabled: bool, admin: User = Depends(admin_user), db: AsyncSession = Depends(get_db)) -> ProviderView:
    provider = await db.get(Provider, provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="provider not found")
    provider.enabled = enabled
    db.add(AuditLog(actor_user_id=admin.id, action="provider.enabled", target_type="provider", target_id=str(provider.id), details={"enabled": enabled}))
    await db.commit()
    await db.refresh(provider)
    return provider_view(provider)


@router.post("/users/{user_id}/credits")
async def grant_credits(user_id: uuid.UUID, payload: AdminCreditGrant, admin: User = Depends(admin_user), db: AsyncSession = Depends(get_db)) -> dict:
    user = await grant(db, user_id, payload.credits, "admin_grant", payload.note)
    db.add(AuditLog(actor_user_id=admin.id, action="credits.grant", target_type="user", target_id=str(user_id), details={"credits": payload.credits, "note": payload.note}))
    await db.commit()
    return {"balance": user.credit_balance, "reserved": user.credit_reserved, "available": user.available_credits}
