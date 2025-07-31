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
echo "Testing numpy import..."
python3 -c "import numpy; print('numpy imported successfully')" || {
    echo "Error: numpy import failed"
    exit 1
}
echo "Testing sentence_transformers import..."
python3 -c "import sentence_transformers; print('sentence_transformers imported successfully')" || {
    echo "Error: sentence_transformers import failed"
    exit 1
}

# Start the application with gunicorn
echo "Starting application..."
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT production_api:app 