from types import SimpleNamespace

from app.provider_router import health_score


def provider(success, failure, latency, priority=100):
    return SimpleNamespace(success_count=success, failure_count=failure, avg_latency_ms=latency, priority=priority)


def test_reliable_fast_provider_scores_higher():
    assert health_score(provider(20, 1, 900, 50)) > health_score(provider(2, 8, 6500, 200))
