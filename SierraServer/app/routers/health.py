from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Provider

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict:
    return {"ok": True, "service": "sierra-vision-server"}


@router.get("/health/ready")
async def ready(db: AsyncSession = Depends(get_db)) -> dict:
    try:
        await db.execute(text("SELECT 1"))
        enabled = await db.scalar(select(func.count()).select_from(Provider).where(Provider.enabled.is_(True))) or 0
        return {"ok": True, "database": True, "enabled_providers": enabled, "cloud_ready": enabled > 0}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
