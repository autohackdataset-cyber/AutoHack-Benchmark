FROM python:3.11.9-slim

WORKDIR /artifact

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default: run full pipeline via entrypoint script
ENTRYPOINT ["bash", "run_all.sh"]
