#!/bin/bash
# Start the FastAPI backend
cd "$(dirname "$0")/backend"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate venv
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "Starting HF-SCF Calculator backend on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
uvicorn main:app --reload --port 8000
