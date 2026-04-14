FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY dimensionalbase/ ./dimensionalbase/

# Install with server, security, and local embeddings
RUN pip install --no-cache-dir ".[server,security,embeddings-local]"

# Expose the default port
EXPOSE 8420

# Environment variables
ENV DMB_BACKEND=sqlite
ENV DMB_DB_PATH=/data/dimensionalbase.db

# Create data directory
RUN mkdir -p /data

# Run the server
CMD ["python", "-m", "dimensionalbase.server", "--host", "0.0.0.0", "--port", "8420", "--db-path", "/data/dimensionalbase.db"]
