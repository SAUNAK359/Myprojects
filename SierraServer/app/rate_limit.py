import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Small single-process limiter for the laptop deployment.

    Sierra runs one Uvicorn worker by design because it also owns the durable job-worker loop.
    Reverse-proxy/provider limits still apply independently; this layer protects auth/job endpoints
    from accidental bursts and basic credential stuffing without introducing another service.
    """

    def __init__(self, app):
        super().__init__(app)
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._requests = 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rule = self._rule(request)
        if rule is not None:
            bucket, maximum, window_seconds = rule
            client = self._client_id(request)
            now = time.monotonic()
            events = self._events[(bucket, client)]
            cutoff = now - window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= maximum:
                retry_after = max(1, int(events[0] + window_seconds - now) + 1)
                return ORJSONResponse(
                    status_code=429,
                    content={"detail": "too many requests; try again shortly"},
                    headers={"Retry-After": str(retry_after)},
                )
            events.append(now)

        self._requests += 1
        if self._requests % 512 == 0:
            self._prune()
        return await call_next(request)

    def _rule(self, request: Request) -> tuple[str, int, int] | None:
        path = request.url.path
        method = request.method.upper()
        if method == "POST" and path in {"/v1/auth/login", "/v1/auth/signup", "/v1/auth/refresh"}:
            return "auth", 15, 60
        if method == "POST" and path == "/v1/jobs":
            return "job_create", 30, 60
        if path.startswith("/admin/") and path != "/admin/login":
            return "admin", 180, 60
        return None

    def _client_id(self, request: Request) -> str:
        return (
            request.headers.get("cf-connecting-ip")
            or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

    def _prune(self) -> None:
        now = time.monotonic()
        stale: list[tuple[str, str]] = []
        for key, values in self._events.items():
            # Longest configured window is one minute. Drop completely idle buckets.
            while values and values[0] <= now - 60:
                values.popleft()
            if not values:
                stale.append(key)
        for key in stale:
            self._events.pop(key, None)
