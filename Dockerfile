# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Install the hospital_triage package
RUN pip install -e .

# Expose port (Hugging Face default is 7860)
EXPOSE 7860

# Ensure logs are visible in HF console
ENV PYTHONUNBUFFERED=1

# Run the full MediFlow Dashboard
CMD ["python", "hospital_api.py"]
