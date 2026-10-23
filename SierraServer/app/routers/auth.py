from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.credits import grant
from app.db import get_db
from app.models import RefreshToken, User
from app.schemas import LoginRequest, LogoutRequest, RefreshRequest, SignupRequest, TokenPair, UserView
from app.security import create_access_token, hash_password, hash_token, new_refresh_token, normalize_email, normalize_phone, refresh_expiry, verify_password

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def user_view(user: User) -> UserView:
    return UserView(
        id=user.id,
        display_name=user.display_name,
        email=user.email,
        phone=user.phone,
        role=user.role,
        credit_balance=user.credit_balance,
        credit_reserved=user.credit_reserved,
        available_credits=user.available_credits,
        created_at=user.created_at,
    )


def _age(dob: date) -> int:
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


async def _issue_pair(db: AsyncSession, user: User, request: Request) -> TokenPair:
    raw_refresh, token_hash = new_refresh_token()
    db.add(RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=refresh_expiry(),
        user_agent=request.headers.get("user-agent", "")[:300],
        ip=(request.client.host if request.client else None),
    ))
    await db.commit()
    return TokenPair(
        access_token=create_access_token(user.id, user.role),
        refresh_token=raw_refresh,
        access_expires_in_seconds=settings.access_token_minutes * 60,
        user=user_view(user),
    )


@router.post("/signup", response_model=TokenPair, status_code=201)
async def signup(payload: SignupRequest, request: Request, db: AsyncSession = Depends(get_db)) -> TokenPair:
    if _age(payload.dob) < settings.min_user_age:
        raise HTTPException(status_code=422, detail=f"Sierra Vision accounts currently require age {settings.min_user_age}+")
    email = normalize_email(str(payload.email))
    try:
        phone = normalize_phone(payload.phone)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    existing = await db.scalar(select(User.id).where(or_(User.email == email, User.phone == phone)))
    if existing:
        raise HTTPException(status_code=409, detail="email or phone is already registered")
    user = User(
        display_name=payload.display_name.strip(),
        email=email,
        phone=phone,
        dob=payload.dob,
        password_hash=hash_password(payload.password),
        credit_balance=0,
        credit_reserved=0,
    )
    db.add(user)
    try:
        await db.flush()
        if settings.trial_credits:
            await grant(db, user.id, settings.trial_credits, "trial_grant", "new account trial")
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="email or phone is already registered") from exc
    await db.refresh(user)
    return await _issue_pair(db, user, request)


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)) -> TokenPair:
    email = normalize_email(str(payload.email))
    user = await db.scalar(select(User).where(User.email == email))
    now = datetime.now(timezone.utc)
    if user and user.locked_until and user.locked_until > now:
        raise HTTPException(status_code=429, detail="account temporarily locked; try again later")
    valid = bool(user and user.is_active and verify_password(payload.password, user.password_hash))
    if not valid:
        if user:
            user.failed_login_count += 1
            if user.failed_login_count >= 5:
                user.locked_until = now + timedelta(minutes=15)
                user.failed_login_count = 0
            await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid email or password")
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now
    await db.commit()
    return await _issue_pair(db, user, request)


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, request: Request, db: AsyncSession = Depends(get_db)) -> TokenPair:
    token = await db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(payload.refresh_token)).with_for_update())
    now = datetime.now(timezone.utc)
    if token is None or token.revoked_at is not None or token.expires_at <= now:
        raise HTTPException(status_code=401, detail="invalid or expired refresh token")
    user = await db.get(User, token.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="account unavailable")
    token.revoked_at = now
    await db.commit()
    return await _issue_pair(db, user, request)


@router.post("/logout", status_code=204)
async def logout(payload: LogoutRequest, db: AsyncSession = Depends(get_db)) -> None:
    token = await db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(payload.refresh_token)))
    if token and token.revoked_at is None:
        token.revoked_at = datetime.now(timezone.utc)
        await db.commit()
