#!/usr/bin/env python3
"""Create test user script."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.database import async_session_factory
from app.core.security import get_password_hash
from app.models.user import User


async def create_test_user():
    """Create a test user."""
    async with async_session_factory() as db:
        # Check if user already exists
        result = await db.execute(
            select(User).where(User.email == "test@example.com")
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            print("Test user already exists:")
            print(f"  Email: {existing_user.email}")
            print(f"  Username: {existing_user.username}")
            return

        # Create test user
        user = User(
            email="test@example.com",
            username="testuser",
            password_hash=get_password_hash("password123"),
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        print("Test user created successfully!")
        print(f"  Email: {user.email}")
        print(f"  Username: {user.username}")
        print(f"  Password: password123")


if __name__ == "__main__":
    asyncio.run(create_test_user())
