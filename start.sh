#!/bin/bash
# Backend startup script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Change to script directory
cd "$(dirname "$0")"

echo -e "${GREEN}Starting Read-Rhyme Backend...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env with your configuration before running again.${NC}"
    exit 1
fi

# Create data directory
mkdir -p data

# Initialize database
echo -e "${YELLOW}Initializing database...${NC}"
python scripts/init_db.py

# Create test user
echo -e "${YELLOW}Creating test user...${NC}"
python scripts/create_test_user.py

# Start server
echo -e "${GREEN}Starting FastAPI server...${NC}"
echo -e "${GREEN}API Documentation: http://localhost:8000/docs${NC}"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
