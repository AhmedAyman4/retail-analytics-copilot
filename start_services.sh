#!/bin/bash

# 1. Start Ollama Server in the background
echo "Starting Ollama Server..."
ollama serve &

# Save PID
SERVER_PID=$!

# 2. Wait for Ollama to be responsive
echo "Waiting for Ollama API..."
until curl -s http://localhost:11434 > /dev/null; do
    sleep 2
    echo "Waiting..."
done

# 3. Pull the requested model
echo "Pulling model phi3.5:3.8b (this may take a while)..."
ollama pull phi3.5:3.8b

# 4. Start Streamlit (Foreground process)
echo "Starting Streamlit App..."
streamlit run app.py --server.address=0.0.0.0 --server.port=7860

# Cleanup
kill $SERVER_PID