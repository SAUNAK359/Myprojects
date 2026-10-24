from app.job_worker import TERMINAL


def test_terminal_states_cover_all_end_states():
    assert TERMINAL == {"completed", "failed", "cancelled", "expired"}


def test_cancelled_is_terminal_and_must_not_be_resurrected():
    assert "cancelled" in TERMINAL


def test_ready_waits_for_capture_settlement():
    assert "ready" not in TERMINAL
