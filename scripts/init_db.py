#!/usr/bin/env python3
"""Database initialization script."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, drop_db
from app.models import *  # noqa: F401, F403


async def main():
    """Initialize database."""
    import argparse

    parser = argparse.ArgumentParser(description="Database management")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate database")
    args = parser.parse_args()

    if args.reset:
        print("Dropping existing database...")
        await drop_db()
        print("Database dropped.")

    print("Initializing database...")
    await init_db()
    print("Database initialized successfully!")
    print("\nTables created:")
    from app.models import Base
    for table in Base.metadata.sorted_tables:
        print(f"  - {table}")


if __name__ == "__main__":
    asyncio.run(main())
