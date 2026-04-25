# Use a slim Python image for efficiency
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (required for some ML libraries and matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY minimal_requirements.txt .
RUN pip install --no-cache-dir -r minimal_requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
# Use the shell form to allow environment variable expansion (like $PORT in Cloud Run)
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
