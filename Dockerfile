# Use a Python base image (3.10 is stable)
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install system dependencies (curl for downloading DB)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data docs agent

# Download the Northwind Database
# We do this in the build step so it's ready when the app starts
RUN curl -L -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Copy the rest of the application
COPY . .

# Set permissions for Hugging Face Spaces (they often run as user 1000)
RUN chown -R 1000:1000 /code

# Switch to non-root user
USER 1000

# Expose the port Streamlit runs on
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]