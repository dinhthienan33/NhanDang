# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create required directories
RUN mkdir -p /app/checkpoint
RUN mkdir -p /app/temp_images
RUN mkdir -p /app/results

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create placeholder file for model
RUN touch /app/checkpoint/model_epoch600.pth

# Copy the backend code
COPY . .

# Make port 8000 available
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 