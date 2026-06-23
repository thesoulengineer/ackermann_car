# Use Python 3.10 slim image as base
FROM python:3.10-slim

LABEL description="Ackermann Car MPC Simulation over ZeroMQ"

# Set working directory inside container
WORKDIR /app

# Install system dependencies for headless rendering (osmesa, GL)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libosmesa6-dev \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Use osmesa for headless OpenGL (required by MuJoCo in a container)
ENV MUJOCO_GL=osmesa

# Copy and install Python dependencies first (for Docker layer caching)
# We install directly from requirements.txt to avoid hatchling wheel-build issues
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project source code
COPY . .

# Make sure all project modules are importable from /app
ENV PYTHONPATH="/app"

# Default: run tests then launch the full simulation (both nodes spawned in run.py)
CMD ["sh", "-c", "pytest && python scripts/run.py"]