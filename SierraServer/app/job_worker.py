import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.credits import release_job_credit
from app.db import SessionLocal
from app.models import Job
from app.provider_router import route_plan

TERMINAL = {"completed", "failed", "cancelled", "expired"}


@dataclass(frozen=True)
class JobAnalysisInput:
    image_path: Path
    camera_context_json: str | None
    device_capabilities_json: str | None
    local_quality_json: str | None
    skill_context: str | None


async def _claim_job() -> uuid.UUID | None:
    async with SessionLocal() as db:
        async with db.begin():
            job = await db.scalar(
                select(Job)
                .where(Job.state == "queued")
                .order_by(Job.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            job.state = "processing"
            job.attempt_count += 1
            job.updated_at = datetime.now(timezone.utc)
            return job.id


async def _job_analysis_input(job_id: uuid.UUID) -> JobAnalysisInput | None:
    async with SessionLocal() as db:
        job = await db.get(Job, job_id)
        if job is None or job.state != "processing" or not job.analysis_image_path:
            return None
        return JobAnalysisInput(
            image_path=Path(job.analysis_image_path),
            camera_context_json=job.camera_context_json,
            device_capabilities_json=job.device_capabilities_json,
            local_quality_json=job.local_quality_json,
            skill_context=job.skill_context,
        )


async def _mark_ready(
    job_id: uuid.UUID, plan: dict, provider_name: str, provider_model: str
) -> bool:
    async with SessionLocal() as db:
        async with db.begin():
            job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if job is None or job.state != "processing":
                return False
            job.plan = plan
            job.provider_name = provider_name
            job.provider_model = provider_model
            job.state = "ready"
            job.ready_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=settings.job_ready_minutes
            )
            job.error_code = None
            job.error_message = None
            return True


async def _mark_failed(job_id: uuid.UUID, exc: Exception) -> None:
    async with SessionLocal() as db:
        async with db.begin():
            job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if job is None or job.state != "processing":
                return
            job.state = "failed"
            job.error_code = "provider_failure"
            job.error_message = str(exc)[:300]
            await release_job_credit(db, job, "analysis failed")


async def _clear_analysis_material(job_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        async with db.begin():
            job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if job is not None:
                job.analysis_image_path = None
                job.camera_context_json = None
                job.device_capabilities_json = None
                job.local_quality_json = None
                job.skill_context = None


async def _process(job_id: uuid.UUID) -> None:
    analysis = await _job_analysis_input(job_id)
    if analysis is None:
        return
    try:
        if not analysis.image_path.is_file():
            raise RuntimeError("analysis image is unavailable")
        image_bytes = analysis.image_path.read_bytes()
        async with SessionLocal() as provider_db:
            plan, provider = await route_plan(
                provider_db,
                image_bytes,
                camera_context_json=analysis.camera_context_json,
                device_capabilities_json=analysis.device_capabilities_json,
                local_quality_json=analysis.local_quality_json,
                skill_context=analysis.skill_context,
            )
            provider_name = provider.name
            provider_model = provider.model
        await _mark_ready(job_id, plan.model_dump(mode="json"), provider_name, provider_model)
    except Exception as exc:
        await _mark_failed(job_id, exc)
    finally:
        try:
            analysis.image_path.unlink(missing_ok=True)
        except Exception:
            pass
        await _clear_analysis_material(job_id)


async def _expire_jobs() -> None:
    now = datetime.now(timezone.utc)
    async with SessionLocal() as db:
        async with db.begin():
            jobs = list(
                (
                    await db.scalars(
                        select(Job)
                        .where(Job.state == "ready", Job.ready_expires_at < now)
                        .order_by(Job.ready_expires_at)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for job in jobs:
                job.state = "expired"
                await release_job_credit(db, job, "ready job expired")


async def recover_stale_processing() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=2)
    async with SessionLocal() as db:
        async with db.begin():
            jobs = list(
                (
                    await db.scalars(
                        select(Job)
                        .where(Job.state == "processing", Job.updated_at < cutoff)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for job in jobs:
                path_ok = bool(job.analysis_image_path and Path(job.analysis_image_path).is_file())
                if path_ok:
                    job.state = "queued"
                else:
                    job.state = "failed"
                    job.error_code = "restart_recovery"
                    job.error_message = "analysis frame was unavailable after server restart"
                    job.camera_context_json = None
                    job.device_capabilities_json = None
                    job.local_quality_json = None
                    job.skill_context = None
                    await release_job_credit(db, job, "restart recovery")


async def worker_loop(stop: asyncio.Event) -> None:
    await recover_stale_processing()
    ticks = 0
    while not stop.is_set():
        job_id = await _claim_job()
        if job_id:
            await _process(job_id)
            continue
        ticks += 1
        if ticks % 20 == 0:
            await _expire_jobs()
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass
