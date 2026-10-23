from fastapi import APIRouter, Depends

from app.deps import current_user
from app.models import User
from app.routers.auth import user_view
from app.schemas import CreditView, UserView

router = APIRouter(prefix="/v1/me", tags=["me"])


@router.get("", response_model=UserView)
async def me(user: User = Depends(current_user)) -> UserView:
    return user_view(user)


@router.get("/credits", response_model=CreditView)
async def credits(user: User = Depends(current_user)) -> CreditView:
    return CreditView(balance=user.credit_balance, reserved=user.credit_reserved, available=user.available_credits)
