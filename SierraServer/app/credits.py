import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import CreditTransaction, Job, User


async def _locked_user(db: AsyncSession, user_id: uuid.UUID) -> User:
    user = await db.scalar(select(User).where(User.id == user_id).with_for_update())
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user


async def ledger(db: AsyncSession, user: User, kind: str, *, job_id: uuid.UUID | None = None, delta_balance: int = 0, delta_reserved: int = 0, note: str | None = None) -> None:
    db.add(CreditTransaction(
        user_id=user.id,
        job_id=job_id,
        kind=kind,
        delta_balance=delta_balance,
        delta_reserved=delta_reserved,
        balance_after=user.credit_balance,
        reserved_after=user.credit_reserved,
        note=note,
    ))


async def grant(db: AsyncSession, user_id: uuid.UUID, amount: int, kind: str, note: str | None = None) -> User:
    user = await _locked_user(db, user_id)
    user.credit_balance += amount
    await ledger(db, user, kind, delta_balance=amount, note=note)
    return user


async def reserve_for_job(db: AsyncSession, user_id: uuid.UUID, job: Job) -> User:
    user = await _locked_user(db, user_id)
    if user.available_credits < 1:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="no Sierra AI credits available")
    user.credit_reserved += 1
    job.reserved_credit = True
    await ledger(db, user, "reserve", job_id=job.id, delta_reserved=1)
    return user


async def commit_job_credit(db: AsyncSession, job: Job) -> User:
    user = await _locked_user(db, job.user_id)
    if not job.reserved_credit:
        return user
    if user.credit_reserved < 1 or user.credit_balance < 1:
        raise RuntimeError("credit ledger invariant violated")
    user.credit_reserved -= 1
    user.credit_balance -= 1
    job.reserved_credit = False
    await ledger(db, user, "commit", job_id=job.id, delta_balance=-1, delta_reserved=-1)
    return user


async def release_job_credit(db: AsyncSession, job: Job, reason: str) -> User:
    user = await _locked_user(db, job.user_id)
    if not job.reserved_credit:
        return user
    user.credit_reserved = max(0, user.credit_reserved - 1)
    job.reserved_credit = False
    await ledger(db, user, "release", job_id=job.id, delta_reserved=-1, note=reason)
    return user
