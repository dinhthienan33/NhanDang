# Automatic Rain Streak Removal System

This project implements a web application for removing rain streaks from images using a deep learning model accelerated with TensorRT and Triton Inference Server.

## Project Structure

- `frontend/`: React.js user interface for image upload and display.
- `backend/`: FastAPI application handling requests, interacting with Kafka/Redis.
- `inference_worker/`: Kafka consumer that triggers inference via Triton.
- `model_repository/`: Stores the optimized model for Triton.
- `scripts/`: Utility scripts (e.g., model conversion).
- `docker-compose.yml`: Docker Compose configuration for local deployment.

## Requirements Met (from @project-rules.mdc)

1.  **Training/Inference Notebook:** Uses the model from the provided Raindrop-Removal repository.
2.  **Front-End:** Simple and responsive UI built with HTML, CSS, and JavaScript.
3.  **Back-End/Pipeline:** FastAPI backend, Kafka for message processing, and Triton for inference.
4.  **Model Storage:** Uses Triton's model repository format with TensorRT optimization.
5.  **Optimization:** Kafka for asynchronous processing and Redis for caching.
6.  **Deployment:** Complete Docker Compose setup for easy deployment.

## Setup & Run (Local using Docker Compose)

### Prerequisites

- Docker and Docker Compose
- NVIDIA Container Toolkit (for Triton GPU support)
- GPU with CUDA support (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd rain-streak-removal
```

### Step 2: Convert the Model to TensorRT Format

```bash
python scripts/convert_to_tensorrt.py \
  --model-path model-checkpoint/model_epoch600.pth \
  --output-dir model_repository \
  --model-name rain_removal \
  --precision fp16
```

### Step 3: Build and Start the System

```bash
docker-compose up --build -d
```

### Step 4: Access the Application

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Usage

1. Open the web application at [http://localhost:3000](http://localhost:3000)
2. Navigate to the "Try It" section
3. Upload an image with rain streaks
4. Click "Process Image" to remove the rain streaks
5. View and download the processed image

## System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Frontend   │────▶│  Backend    │────▶│   Kafka     │────▶│   Worker    │
│  (Nginx)    │     │  (FastAPI)  │     │             │     │             │
│             │     │             │     │             │     │             │
└─────────────┘     └──────┬──────┘     └─────────────┘     └──────┬──────┘
                           │                                        │
                           │                                        │
                           ▼                                        ▼
                    ┌─────────────┐                         ┌─────────────┐
                    │             │                         │             │
                    │    Redis    │                         │   Triton    │
                    │  (Caching)  │                         │  Inference  │
                    │             │                         │   Server    │
                    └─────────────┘                         └─────────────┘
```

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, FastAPI
- **Message Queue**: Kafka
- **Caching**: Redis
- **Inference**: NVIDIA Triton Inference Server, TensorRT
- **Containerization**: Docker, Docker Compose

## Development

### Running Individual Components

Each component can be built and run separately for development:

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
# Any static file server will work, for example:
python -m http.server 3000
```

## Troubleshooting

### Common Issues

1. **GPU Access Issues**: Make sure the NVIDIA Container Toolkit is properly installed and configured.
2. **Model Conversion Errors**: Check that the model path is correct and the model format is supported.
3. **Kafka Connection Errors**: Ensure Zookeeper is running before Kafka.

## License

[MIT License](LICENSE)
