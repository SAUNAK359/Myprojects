import asyncio
import getpass
from datetime import date

from sqlalchemy import or_, select

from app.db import SessionLocal
from app.models import User
from app.security import hash_password, normalize_email, normalize_phone


def prompt_nonempty(label: str) -> str:
    while True:
        value = input(label).strip()
        if value:
            return value
        print("Value is required.")


def prompt_dob() -> date:
    while True:
        raw = input("Admin DOB (YYYY-MM-DD): ").strip()
        try:
            return date.fromisoformat(raw)
        except ValueError:
            print("Invalid date. Use YYYY-MM-DD.")


def prompt_password() -> str:
    while True:
        first = getpass.getpass("Admin password (minimum 16 characters): ")
        if len(first) < 16:
            print("Password must be at least 16 characters.")
            continue
        second = getpass.getpass("Confirm admin password: ")
        if first != second:
            print("Passwords do not match.")
            continue
        return first


async def main() -> None:
    print("Sierra Vision secure admin bootstrap")
    print("The password is hashed immediately and is never written to .env.")
    print()

    email = normalize_email(prompt_nonempty("Admin email: "))
    phone = normalize_phone(prompt_nonempty("Admin phone with country code: "))
    dob = prompt_dob()
    display_name = input("Display name [Sierra Administrator]: ").strip() or "Sierra Administrator"
    password = prompt_password()

    async with SessionLocal() as db:
        existing = await db.scalar(select(User).where(or_(User.email == email, User.phone == phone)))
        if existing:
            if existing.role == "admin" and existing.email == email and existing.phone == phone:
                raise SystemExit("An admin account with that email/phone already exists.")
            raise SystemExit("Email or phone already belongs to another account; refusing privilege escalation.")

        user = User(
            email=email,
            phone=phone,
            display_name=display_name,
            dob=dob,
            password_hash=hash_password(password),
            role="admin",
            credit_balance=0,
            credit_reserved=0,
        )
        db.add(user)
        await db.commit()
        print("Admin account created successfully.")


if __name__ == "__main__":
    asyncio.run(main())
