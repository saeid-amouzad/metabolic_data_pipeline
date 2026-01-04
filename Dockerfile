# ------------------------------------------------------------
# Dockerfile for Metabolic Data Pipeline
# ------------------------------------------------------------

# 1. Base image with Python
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Ensure project root is on PYTHONPATH
ENV PYTHONPATH=/app

# 4. Copy dependency definitions first (for Docker cache)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the full project into the container
COPY . .

# 7. Create input_data directory (user-facing input mount point)
#    This directory is intended for user-provided raw data that will be
#    uploaded via the dashboard UI.
RUN mkdir -p /app/input_data

# 8. Expose Flask port
EXPOSE 5000

# 9. Run the Flask dashboard
CMD ["python", "dashboard/app.py"]
