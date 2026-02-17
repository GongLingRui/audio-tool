"""Authentication API routes."""
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.deps import DbDep, CurrentUserDep
from app.core.exceptions import BadRequestException, UnauthorizedException
from app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
    decode_access_token as decode_token,
)
from app.config import settings
from app.models.user import User as UserModel
from app.schemas.common import ApiResponse, ErrorDetail
from app.schemas.user import Token, User, UserCreate, UserLogin

router = APIRouter()


@router.post("/register", response_model=ApiResponse[Token])
async def register(user_data: UserCreate, db: DbDep):
    """Register a new user."""
    # Check if email already exists
    result = await db.execute(select(UserModel).where(UserModel.email == user_data.email))
    if result.scalar_one_or_none():
        raise BadRequestException("Email already registered")

    # Check if username already exists
    result = await db.execute(select(UserModel).where(UserModel.username == user_data.username))
    if result.scalar_one_or_none():
        raise BadRequestException("Username already taken")

    # Create new user
    user = UserModel(
        email=user_data.email,
        username=user_data.username,
        password_hash=get_password_hash(user_data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Generate token
    access_token = create_access_token(
        subject=user.id,
        expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
    )

    return ApiResponse(
        data=Token(
            access_token=access_token,
            user=User.model_validate(user),
        )
    )


logger = logging.getLogger(__name__)


@router.post("/login", response_model=ApiResponse[Token])
async def login(credentials: UserLogin, db: DbDep):
    """Login user."""
    try:
        # Find user by email
        result = await db.execute(select(UserModel).where(UserModel.email == credentials.email))
        user = result.scalar_one_or_none()

        if not user or not verify_password(credentials.password, user.password_hash):
            raise UnauthorizedException("Invalid email or password")

        if not user.is_active:
            raise UnauthorizedException("User account is disabled")

        # Generate token (exp 使用整数时间戳，避免 jose 编码异常)
        access_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
        )

        return ApiResponse(
            data=Token(
                access_token=access_token,
                user=User.model_validate(user),
            )
        )
    except (UnauthorizedException, BadRequestException):
        raise
    except Exception as e:
        logger.exception("Login failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e) if settings.debug else "登录失败，请稍后重试",
        )


@router.post("/refresh", response_model=ApiResponse[Token])
async def refresh_token(
    authorization: str = Header(..., description="Authorization header with bearer token"),
    db: DbDep = None,
):
    """Refresh access token using a valid existing token."""
    # Extract token from Authorization header
    if not authorization.startswith("Bearer "):
        raise UnauthorizedException("Invalid authorization header format")

    token = authorization[7:]  # Remove "Bearer " prefix

    # Decode and verify the existing token
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise UnauthorizedException("Invalid token: missing user ID")

        # Get user from database
        result = await db.execute(select(UserModel).where(UserModel.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise UnauthorizedException("User not found")

        if not user.is_active:
            raise UnauthorizedException("User account is disabled")

        # Generate new access token
        access_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
        )

        return ApiResponse(
            data=Token(
                access_token=access_token,
                user=User.model_validate(user),
            )
        )

    except Exception as e:
        raise UnauthorizedException(f"Invalid token: {str(e)}")


@router.get("/me", response_model=ApiResponse[User])
async def get_current_user_info(
    current_user: CurrentUserDep,
):
    """Get current user information."""
    return ApiResponse(
        data=User.model_validate(current_user)
    )
