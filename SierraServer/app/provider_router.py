from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Provider
from app.provider_client import call_provider
from app.schemas import SierraPlan


def health_score(provider: Provider) -> float:
    reliability = (provider.success_count + 1) / (provider.success_count + provider.failure_count + 2)
    speed = 0.55 if provider.avg_latency_ms <= 0 else max(0.0, 1.0 - min(provider.avg_latency_ms, 10000) / 10000)
    priority = max(0.0, 1.0 - min(provider.priority, 1000) / 1000)
    return reliability * 0.62 + speed * 0.28 + priority * 0.10


async def route_plan(db: AsyncSession, image_bytes: bytes) -> tuple[SierraPlan, Provider]:
    providers = list((await db.scalars(select(Provider).where(Provider.enabled.is_(True)))).all())
    if not providers:
        raise RuntimeError("no Sierra AI provider is enabled")
    providers.sort(key=health_score, reverse=True)
    last_error: Exception | None = None
    for provider in providers:
        try:
            plan, latency = await call_provider(provider, image_bytes)
            provider.success_count += 1
            provider.avg_latency_ms = latency if provider.avg_latency_ms <= 0 else int((provider.avg_latency_ms * 3 + latency) / 4)
            provider.last_error = None
            await db.commit()
            return plan, provider
        except Exception as exc:
            provider.failure_count += 1
            provider.last_error = str(exc)[:300]
            await db.commit()
            last_error = exc
    raise RuntimeError(f"all Sierra AI providers failed: {last_error}")
