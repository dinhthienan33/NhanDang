# Automatic Rain Streak Removal System

This project implements a web application for removing rain streaks from images using a deep learning model accelerated with TensorRT and Triton Inference Server.

## Project Structure

- `frontend/`: HTML/CSS/JavaScript user interface for image upload and display.
- `backend/`: FastAPI application handling requests, interacting with Redis.
- `inference_worker/`: Kafka consumer that triggers inference (for production deployment).
- `model_repository/`: Stores the optimized model for Triton (for production deployment).
- `scripts/`: Utility scripts (e.g., model conversion).
- `docker-compose.yml`: Docker Compose configuration for local deployment.

## Requirements Met

1.  **Training/Inference Model:** Uses the model from the provided Raindrop-Removal repository.
2.  **Front-End:** Simple and responsive UI built with HTML, CSS, and JavaScript.
3.  **Back-End/Pipeline:** FastAPI backend with optional Kafka/Triton integration.
4.  **Model Storage:** For production, uses Triton's model repository format with TensorRT optimization.
5.  **Optimization:** Redis for caching, with optional Kafka for async processing in production.
6.  **Deployment:** Docker Compose setup for easy deployment.

## Quick Start (Testing/Development)

This setup uses a mock model implementation for testing so you don't need the actual model or GPU.

### Prerequisites

- Docker and Docker Compose

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd rain-streak-removal
```

### Step 2: Start the Development System

```bash
docker-compose up --build
```

This starts:
- Frontend on http://localhost:3000
- Backend on http://localhost:8000 (using mock model)
- Redis for caching

### Step 3: Test the Application

1. Open the web application at http://localhost:3000
2. Upload an image with rain streaks
3. Click "Process Image" to see a simulated de-raining effect
4. View and download the processed image

## Production Deployment with Real Model

For production with the actual model and TensorRT acceleration:

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

### Step 1: Prepare the Model

If you have the PyTorch model:

```bash
# Create model repository structure
python scripts/convert_to_tensorrt.py \
  --model-path model-checkpoint/model_epoch600.pth \
  --output-dir model_repository \
  --model-name rain_removal \
  --precision fp16
```

### Step 2: Edit docker-compose.yml

1. Uncomment the Triton, Kafka, Zookeeper, and Worker services
2. Set `USE_MOCK_MODEL=false` in the backend service

### Step 3: Start the System

```bash
docker-compose up --build -d
```

## API Documentation

When the system is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Frontend Can't Connect to Backend

Check CORS settings in the backend and make sure the URL is correct in the frontend JavaScript.

### Mock Model vs Real Model

- The system defaults to using a mock implementation that applies basic image processing
- Set `USE_MOCK_MODEL=false` in docker-compose.yml to use the real neural network model
- Mock results will have a "MOCK RESULT" watermark to clearly indicate they're not from the real model

### Model Loading Errors

If you see model loading errors:

1. Check that the model file exists in the expected location
2. Verify the model format is compatible
3. Make sure PyTorch and other dependencies are installed
4. For GPU errors, verify CUDA is working properly

## Development

### Running Backend Locally

```bash
cd backend
pip install -r requirements.txt
export USE_MOCK_MODEL=true  # Use mock model for testing
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running Frontend Locally

```bash
cd frontend
# Use any static file server, for example:
python -m http.server 3000
```

## Extending the System

### Adding More ML Models

1. Create new model wrapper classes similar to `RainRemovalModel`
2. Add new endpoints in the FastAPI application
3. Integrate with the frontend

### Scaling with Kubernetes

For production deployment at scale, consider:

1. Converting the docker-compose.yml to Kubernetes manifests
2. Setting up proper resource limits and requests
3. Using Horizontal Pod Autoscaling
4. Implementing proper monitoring and logging

## License

[MIT License](LICENSE)
