from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8)


def _token(user: User) -> dict[str, str]:
    return {"access_token": create_access_token(user.id), "token_type": "bearer"}


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(credentials: Credentials, session: Session = Depends(get_db)) -> dict[str, str]:
    email = str(credentials.email).lower()
    if session.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="email already registered")
    user = User(email=email, password_hash=hash_password(credentials.password))
    session.add(user)
    session.commit()
    return _token(user)


@router.post("/login")
def login(credentials: Credentials, session: Session = Depends(get_db)) -> dict[str, str]:
    user = session.scalar(select(User).where(User.email == str(credentials.email).lower()))
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")
    return _token(user)
