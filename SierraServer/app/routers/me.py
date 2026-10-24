from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import current_user
from app.models import Device, Job, RefreshToken, User, utcnow
from app.routers.auth import user_view
from app.schemas import (
    CreditView,
    DeleteAccountRequest,
    DeviceProfileInput,
    DeviceView,
    PasswordChangeRequest,
    UserView,
)
from app.security import hash_password, verify_password

router = APIRouter(prefix="/v1/me", tags=["me"])


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else None)


def _device_view(device: Device) -> DeviceView:
    return DeviceView(
        id=device.id,
        install_id=device.install_id,
        manufacturer=device.manufacturer,
        model=device.model,
        device_name=device.device_name,
        android_release=device.android_release,
        sdk_int=device.sdk_int,
        app_version=device.app_version,
        app_build=device.app_build,
        locale=device.locale,
        timezone=device.timezone,
        screen_width_px=device.screen_width_px,
        screen_height_px=device.screen_height_px,
        density_dpi=device.density_dpi,
        is_emulator=device.is_emulator,
        sierra_beta_version=device.sierra_beta_version,
        camera_capabilities=device.camera_capabilities,
        first_seen_at=device.first_seen_at,
        last_seen_at=device.last_seen_at,
    )


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


@router.put("/devices/current", response_model=DeviceView)
async def sync_current_device(
    payload: DeviceProfileInput,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> DeviceView:
    device = await db.scalar(
        select(Device).where(Device.user_id == user.id, Device.install_id == payload.install_id)
    )
    if device is None:
        device = Device(user_id=user.id, install_id=payload.install_id)
        db.add(device)

    device.manufacturer = payload.manufacturer.strip()
    device.model = payload.model.strip()
    device.device_name = payload.device_name.strip() if payload.device_name else None
    device.android_release = payload.android_release.strip()
    device.sdk_int = payload.sdk_int
    device.app_version = payload.app_version.strip()
    device.app_build = payload.app_build
    device.locale = payload.locale.strip()
    device.timezone = payload.timezone.strip()
    device.screen_width_px = payload.screen_width_px
    device.screen_height_px = payload.screen_height_px
    device.density_dpi = payload.density_dpi
    device.is_emulator = payload.is_emulator
    device.sierra_beta_version = payload.sierra_beta_version
    if payload.camera_capabilities is not None:
        device.camera_capabilities = payload.camera_capabilities
    device.last_seen_at = utcnow()
    device.last_ip = _client_ip(request)
    device.last_user_agent = request.headers.get("user-agent", "")[:300]
    await db.commit()
    await db.refresh(device)
    return _device_view(device)


@router.get("/devices", response_model=list[DeviceView])
async def list_devices(
    user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
) -> list[DeviceView]:
    devices = list(
        (
            await db.scalars(
                select(Device).where(Device.user_id == user.id).order_by(Device.last_seen_at.desc())
            )
        ).all()
    )
    return [_device_view(device) for device in devices]


@router.post("/password", status_code=204)
async def change_password(
    payload: PasswordChangeRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current password is incorrect",
        )
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
        raise HTTPException(
            status_code=409,
            detail="admin accounts cannot be deleted through the customer endpoint",
        )
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="password is incorrect",
        )

    analysis_paths = list(
        (
            await db.scalars(
                select(Job.analysis_image_path).where(
                    Job.user_id == user.id,
                    Job.analysis_image_path.is_not(None),
                )
            )
        ).all()
    )
    await db.execute(delete(User).where(User.id == user.id))
    await db.commit()

    for value in analysis_paths:
        if value:
            try:
                Path(value).unlink(missing_ok=True)
            except OSError:
                pass
