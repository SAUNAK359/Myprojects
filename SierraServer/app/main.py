import asyncio
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.job_worker import worker_loop
from app.models import User
from app.routers import admin, auth, health, jobs, me
from app.security import hash_password, normalize_email, normalize_phone


async def bootstrap_admin() -> None:
    email = settings.admin_bootstrap_email.strip()
    password = settings.admin_bootstrap_password.get_secret_value()
    if not email or not password:
        return
    if len(password) < 16:
        raise RuntimeError("bootstrap admin password must be at least 16 characters")
    async with SessionLocal() as db:
        existing = await db.scalar(select(User).where(User.email == normalize_email(email)))
        if existing:
            if existing.role != "admin":
                existing.role = "admin"
                await db.commit()
            return
        phone = normalize_phone(settings.admin_bootstrap_phone)
        admin_user = User(
            email=normalize_email(email),
            phone=phone,
            display_name="Sierra Administrator",
            dob=date.fromisoformat(settings.admin_bootstrap_dob),
            password_hash=hash_password(password),
            role="admin",
            credit_balance=0,
            credit_reserved=0,
        )
        db.add(admin_user)
        await db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await bootstrap_admin()
    stop = asyncio.Event()
    task = asyncio.create_task(worker_loop(stop), name="sierra-job-worker")
    try:
        yield
    finally:
        stop.set()
        await task


app = FastAPI(
    title="Sierra Vision Server",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(me.router)
app.include_router(jobs.router)
app.include_router(admin.router)


@app.get("/")
async def root() -> dict:
    return {"service": "Sierra Vision Server", "version": "0.1.0", "docs": "/docs", "health": "/health/ready"}
