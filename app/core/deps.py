"""Dependency injection utilities."""
from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt

from app.config import settings
from app.database import get_db
from app.models.user import User

# auto_error=False: 未携带 token 时由我们在 get_current_user 中返回 401，而不是 FastAPI 默认的 403
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Get current authenticated user.

    Args:
        credentials: HTTP authorization credentials (None if header missing)
        db: Database session

    Returns:
        User: Current user

    Raises:
        HTTPException: 401 if no token or invalid token, 403 if user disabled
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未登录或登录已过期，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise credentials_exception

    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    from sqlalchemy import select

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current active user.

    Args:
        current_user: Current user

    Returns:
        User: Current active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    return current_user


# Type aliases for dependency injection
CurrentUserDep = Annotated[User, Depends(get_current_active_user)]
DbDep = Annotated[AsyncSession, Depends(get_db)]
