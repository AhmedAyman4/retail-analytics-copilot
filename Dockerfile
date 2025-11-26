# Use standard Python image (much lighter than nvidia/cuda)
FROM python:3.10-slim

WORKDIR /code

# 1. Environment Variables
ENV PYTHONPATH=/code
ENV PYTHONUNBUFFERED=1
# Set OLLAMA_HOST so it listens on all interfaces (internal convenience)
ENV OLLAMA_HOST=0.0.0.0

# 2. System Deps & Ollama Install
# We install curl to download ollama script
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Install Python Deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create Directory Structure
RUN mkdir -p data docs agent .ollama

# 5. Download DB
RUN curl -L -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# 6. Copy Code
COPY . .

# 7. Permissions for User 1000 (Hugging Face Spaces requirement)
# We also ensure the .ollama directory is writable by the user
RUN useradd -m -u 1000 user
RUN chown -R 1000:1000 /code
RUN mkdir -p /home/user/.ollama && chown -R 1000:1000 /home/user/.ollama
# Set HOME to ensure ollama writes models to the user's dir
ENV HOME=/home/user

# Copy and enable start script
COPY start_services.sh .
RUN chmod +x start_services.sh

USER 1000

# Expose ports
EXPOSE 7860
EXPOSE 11434

# Entrypoint
CMD ["./start_services.sh"]