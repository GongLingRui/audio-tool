"""User schemas."""
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)


class UserCreate(UserBase):
    """User creation schema."""

    password: str = Field(..., min_length=6, max_length=100)


class UserLogin(BaseModel):
    """User login schema."""

    email: EmailStr
    password: str


class User(UserBase):
    """User response schema."""

    id: str
    avatar_url: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema."""

    access_token: str
    refresh_token: str | None = None
    user: User
