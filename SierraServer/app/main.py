import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.config import prepare_runtime_directories, settings, validate_runtime_config
from app.job_worker import worker_loop
from app.rate_limit import RateLimitMiddleware
from app.routers import admin, auth, health, jobs, me


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_runtime_config(settings)
    prepare_runtime_directories(settings)
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
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Sierra's laptop deployment intentionally runs one Uvicorn worker because that process also
# owns the durable worker loop. The process-local limiter is authoritative for this origin.
app.add_middleware(RateLimitMiddleware)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    if request.url.scheme == "https" or forwarded_proto == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    if request.url.path.startswith(("/v1/auth", "/v1/me", "/admin")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"

    if request.url.path.startswith("/admin"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        )
    return response


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(me.router)
app.include_router(jobs.router)
app.include_router(admin.router)


@app.get("/")
async def root() -> dict:
    result = {
        "service": "Sierra Vision Server",
        "version": "0.1.0",
        "health": "/health/ready",
    }
    if not settings.is_production:
        result["docs"] = "/docs"
        result["admin"] = "/admin/login"
    return result
