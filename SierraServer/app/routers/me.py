from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import current_user
from app.models import RefreshToken, User
from app.routers.auth import user_view
from app.schemas import CreditView, DeleteAccountRequest, PasswordChangeRequest, UserView
from app.security import hash_password, verify_password

router = APIRouter(prefix="/v1/me", tags=["me"])


@router.get("", response_model=UserView)
async def me(user: User = Depends(current_user)) -> UserView:
    return user_view(user)


@router.get("/credits", response_model=CreditView)
async def credits(user: User = Depends(current_user)) -> CreditView:
    return CreditView(
        balance=user.credit_balance,
        reserved=user.credit_reserved,
        available=user.available_credits,
    )


@router.post("/password", status_code=204)
async def change_password(
    payload: PasswordChangeRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="current password is incorrect")
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=422, detail="new password must be different")
    user.password_hash = hash_password(payload.new_password)
    await db.execute(delete(RefreshToken).where(RefreshToken.user_id == user.id))
    await db.commit()


@router.delete("", status_code=204)
async def delete_account(
    payload: DeleteAccountRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    if user.role == "admin":
        raise HTTPException(status_code=409, detail="admin accounts cannot be deleted through the customer endpoint")
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="password is incorrect")
    await db.delete(user)
    await db.commit()
