#!/bin/bash

# 1. Start Ollama Server
echo "Starting Ollama Server..."
ollama serve &
SERVER_PID=$!

# 2. Wait for Ollama API to be up
echo "Waiting for Ollama API..."
until curl -s http://localhost:11434 > /dev/null; do
    sleep 2
    echo "Waiting for localhost:11434..."
done

# 3. Pull Model in BACKGROUND (&)
# This allows Streamlit to start immediately so the Space doesn't crash.
echo "Triggering background pull of phi3.5:3.8b..."
ollama pull phi3.5:3.8b &

# 4. Start Streamlit immediately
# Added --server.enableCORS=false and --server.enableXsrfProtection=false to fix Upload 403 Errors
echo "Starting Streamlit..."
streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

# Cleanup
kill $SERVER_PID