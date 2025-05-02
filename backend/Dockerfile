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

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Raindrop-Removal source code
#COPY Raindrop-Removal/model ./model
#COPY Raindrop-Removal/layers.py .
#COPY Raindrop-Removal/networks.py .
#COPY Raindrop-Removal/utils.py .

# Copy the model checkpoint
COPY /model-checkpoint/model_epoch600.pth ./model-checkpoint/

# Copy the backend code
COPY . .

# Make port 8000 available
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 