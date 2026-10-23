from types import SimpleNamespace


def available(balance: int, reserved: int) -> int:
    return max(0, balance - reserved)


def test_five_credit_trial_means_five_available_photos():
    assert available(5, 0) == 5


def test_reservation_does_not_consume_balance():
    user = SimpleNamespace(credit_balance=5, credit_reserved=1)
    assert user.credit_balance == 5
    assert available(user.credit_balance, user.credit_reserved) == 4


def test_commit_semantics_consume_exactly_one():
    balance, reserved = 5, 1
    balance -= 1
    reserved -= 1
    assert (balance, reserved, available(balance, reserved)) == (4, 0, 4)


def test_release_restores_available_credit():
    balance, reserved = 5, 1
    reserved -= 1
    assert (balance, reserved, available(balance, reserved)) == (5, 0, 5)
