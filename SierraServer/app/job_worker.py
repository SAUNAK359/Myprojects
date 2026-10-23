import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.credits import release_job_credit
from app.db import SessionLocal
from app.models import Job
from app.provider_router import route_plan

TERMINAL = {"completed", "failed", "cancelled", "expired"}


async def _claim_job() -> str | None:
    async with SessionLocal() as db:
        async with db.begin():
            job = await db.scalar(
                select(Job).where(Job.state == "queued").order_by(Job.created_at).limit(1).with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            job.state = "processing"
            job.attempt_count += 1
            job.updated_at = datetime.now(timezone.utc)
            return str(job.id)


async def _process(job_id: str) -> None:
    async with SessionLocal() as db:
        job = await db.get(Job, job_id)
        if job is None or job.state != "processing":
            return
        path = Path(job.analysis_image_path or "")
        try:
            if not path.is_file():
                raise RuntimeError("analysis image is unavailable")
            plan, provider = await route_plan(db, path.read_bytes())
            job.plan = plan.model_dump(mode="json")
            job.provider_name = provider.name
            job.provider_model = provider.model
            job.state = "ready"
            job.ready_expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.job_ready_minutes)
            job.error_code = None
            job.error_message = None
            await db.commit()
        except Exception as exc:
            await db.rollback()
            job = await db.get(Job, job_id)
            if job is not None and job.state == "processing":
                job.state = "failed"
                job.error_code = "provider_failure"
                job.error_message = str(exc)[:300]
                await release_job_credit(db, job, "analysis failed")
                await db.commit()
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            async with SessionLocal() as clean_db:
                clean_job = await clean_db.get(Job, job_id)
                if clean_job and clean_job.analysis_image_path:
                    clean_job.analysis_image_path = None
                    await clean_db.commit()


async def _expire_jobs() -> None:
    now = datetime.now(timezone.utc)
    async with SessionLocal() as db:
        jobs = list((await db.scalars(select(Job).where(Job.state == "ready", Job.ready_expires_at < now))).all())
        for job in jobs:
            job.state = "expired"
            await release_job_credit(db, job, "ready job expired")
        if jobs:
            await db.commit()


async def recover_stale_processing() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=2)
    async with SessionLocal() as db:
        jobs = list((await db.scalars(select(Job).where(Job.state == "processing", Job.updated_at < cutoff))).all())
        for job in jobs:
            path_ok = bool(job.analysis_image_path and Path(job.analysis_image_path).is_file())
            if path_ok:
                job.state = "queued"
            else:
                job.state = "failed"
                job.error_code = "restart_recovery"
                job.error_message = "analysis frame was unavailable after server restart"
                await release_job_credit(db, job, "restart recovery")
        if jobs:
            await db.commit()


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
