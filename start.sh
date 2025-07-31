#!/bin/bash

# Startup script for Render deployment

echo "Starting LLM Document Processing API..."

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY environment variable is not set"
fi

# Create cache directory if it doesn't exist
mkdir -p ./cache

# Test pdfplumber import
echo "Testing pdfplumber import..."
python3 -c "import pdfplumber; print('pdfplumber imported successfully')" || {
    echo "Error: pdfplumber import failed"
    exit 1
}

# Start the application with gunicorn
echo "Starting application..."
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT production_api:app 