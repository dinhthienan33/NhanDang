import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Rain Removal Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the model
MODEL_PATH = os.environ.get('MODEL_PATH', "model-checkpoint/model_epoch600.pth")
USE_MOCK_MODEL = False
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model
    
    try:
        if USE_MOCK_MODEL:
            # Use mock model for testing
            from mock_model import MockRainRemovalModel
            model = MockRainRemovalModel(MODEL_PATH)
        else:
            # Try to load the real model
            try:
                from rain_removal import RainRemovalModel
                model = RainRemovalModel(MODEL_PATH)
            except Exception as e:
                # If real model fails, fall back to mock model
                print(f"Failed to load real model: {e}")
                print("Falling back to mock model...")
                from mock_model import MockRainRemovalModel
                model = MockRainRemovalModel(MODEL_PATH)
    except Exception as e:
        print(f"Error initializing model: {e}")
        # We'll handle this gracefully if someone tries to use the API

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Check if the service is running."""
    return {"status": "ok", "model_loaded": model is not None, "using_mock": USE_MOCK_MODEL}

@app.post("/process", tags=["Rain Removal"])
async def process_image(file: UploadFile = File(...)):
    """Process an image to remove rain streaks."""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
            
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Process image
        result = model.process_image(img)
        
        # Encode result
        _, encoded_img = cv2.imencode('.png', result)
        
        # Return processed image
        return Response(content=encoded_img.tobytes(), 
                       media_type="image/png")
                       
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 