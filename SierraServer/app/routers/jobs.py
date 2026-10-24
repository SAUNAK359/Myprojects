import json
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.credits import commit_job_credit, release_job_credit, reserve_for_job
from app.db import get_db
from app.deps import current_user
from app.models import Job, User
from app.schemas import CompleteJobRequest, CreditView, JobView, SierraPlan

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
TERMINAL = {"completed", "failed", "cancelled", "expired"}
MAX_CONTEXT_CHARS = 96_000
MAX_SKILL_CHARS = 24_000


def _view(job: Job, user: User) -> JobView:
    return JobView(
        id=job.id,
        state=job.state,
        plan=SierraPlan.model_validate(job.plan) if job.plan else None,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        ready_expires_at=job.ready_expires_at,
        credits=CreditView(
            balance=user.credit_balance,
            reserved=user.credit_reserved,
            available=user.available_credits,
        ),
    )


async def _existing_job(db: AsyncSession, user_id: uuid.UUID, key: str) -> Job | None:
    return await db.scalar(select(Job).where(Job.user_id == user_id, Job.idempotency_key == key))


def _validated_json_context(value: str, label: str) -> str:
    clean = value.strip() or "{}"
    if len(clean) > MAX_CONTEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"{label} is too large")
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{label} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail=f"{label} must be a JSON object")
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)


@router.post("", response_model=JobView, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    image: UploadFile = File(...),
    camera_context_json: str = Form("{}"),
    device_capabilities_json: str = Form("{}"),
    local_quality_json: str = Form("{}"),
    skill_context: str = Form(""),
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=8, max_length=100),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> JobView:
    existing = await _existing_job(db, user.id, idempotency_key)
    if existing:
        await db.refresh(user)
        return _view(existing, user)

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="analysis image must be JPEG, PNG or WebP")
    raw = await image.read(settings.max_upload_bytes + 1)
    if not raw or len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="analysis image is too large")

    is_jpeg = raw[:2] == b"\xff\xd8"
    is_png = raw[:8] == b"\x89PNG\r\n\x1a\n"
    is_webp = len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(status_code=415, detail="analysis payload is not a recognized image")

    camera_context = _validated_json_context(camera_context_json, "camera_context_json")
    device_context = _validated_json_context(device_capabilities_json, "device_capabilities_json")
    quality_context = _validated_json_context(local_quality_json, "local_quality_json")
    clean_skill_context = skill_context.strip()
    if len(clean_skill_context) > MAX_SKILL_CHARS:
        raise HTTPException(status_code=413, detail="skill_context is too large")

    job = Job(
        user_id=user.id,
        idempotency_key=idempotency_key,
        state="queued",
        camera_context_json=camera_context,
        device_capabilities_json=device_context,
        local_quality_json=quality_context,
        skill_context=clean_skill_context or None,
    )
    db.add(job)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        winner = await _existing_job(db, user.id, idempotency_key)
        if winner is None:
            raise
        await db.refresh(user)
        return _view(winner, user)

    path = settings.runtime_dir / "analysis" / f"{job.id}_{secrets.token_hex(8)}.img"
    try:
        path.write_bytes(raw)
        job.analysis_image_path = str(path.resolve())
        user = await reserve_for_job(db, user.id, job)
        await db.commit()
    except Exception:
        await db.rollback()
        path.unlink(missing_ok=True)
        raise

    await db.refresh(job)
    return _view(job, user)


@router.get("/{job_id}", response_model=JobView)
async def get_job(
    job_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> JobView:
    job = await db.scalar(select(Job).where(Job.id == job_id, Job.user_id == user.id))
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    await db.refresh(user)
    return _view(job, user)


@router.post("/{job_id}/complete", response_model=JobView)
async def complete_job(
    job_id: uuid.UUID,
    payload: CompleteJobRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> JobView:
    job = await db.scalar(
        select(Job).where(Job.id == job_id, Job.user_id == user.id).with_for_update()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state == "completed":
        if job.capture_id and job.capture_id != payload.capture_id:
            raise HTTPException(status_code=409, detail="job was completed by a different capture")
        await db.refresh(user)
        return _view(job, user)
    if job.state != "ready":
        raise HTTPException(status_code=409, detail=f"job cannot complete from state {job.state}")
    if job.ready_expires_at and job.ready_expires_at <= datetime.now(timezone.utc):
        job.state = "expired"
        await release_job_credit(db, job, "complete arrived after expiry")
        await db.commit()
        raise HTTPException(status_code=409, detail="job expired before final capture confirmation")

    user = await commit_job_credit(db, job)
    job.capture_id = payload.capture_id
    job.state = "completed"
    job.completed_at = datetime.now(timezone.utc)
    job.error_code = None
    job.error_message = None
    await db.commit()
    return _view(job, user)


@router.post("/{job_id}/cancel", response_model=JobView)
async def cancel_job(
    job_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> JobView:
    job = await db.scalar(
        select(Job).where(Job.id == job_id, Job.user_id == user.id).with_for_update()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state in TERMINAL:
        await db.refresh(user)
        return _view(job, user)

    job.state = "cancelled"
    user = await release_job_credit(db, job, "client cancelled")
    image_path = job.analysis_image_path
    job.analysis_image_path = None
    job.camera_context_json = None
    job.device_capabilities_json = None
    job.local_quality_json = None
    job.skill_context = None
    await db.commit()
    if image_path:
        Path(image_path).unlink(missing_ok=True)
    return _view(job, user)
