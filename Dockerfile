# Use a Python base image (3.10 is stable)
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Set environment variables
# 1. Force Python to verify /code is a package root
ENV PYTHONPATH=/code
# 2. Set Hugging Face cache to a directory we can control
ENV HF_HOME=/code/.cache/huggingface
# 3. Standard Python settings
ENV PYTHONUNBUFFERED=1

# Install system dependencies (curl for downloading DB)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories explicitly
# We create the cache directory here so we can chown it later
RUN mkdir -p data docs agent .cache

# Download the Northwind Database
RUN curl -L -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Copy the rest of the application
COPY . .

# CRITICAL FIX: Change ownership of the ENTIRE /code directory to user 1000
# This ensures the user can write to the cache directory we defined above
RUN chown -R 1000:1000 /code

# Switch to non-root user
USER 1000

# Expose the port Streamlit runs on
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]