import os
import cv2  # type: ignore # Ignore cv2 import errors
import numpy as np
import torch  # type: ignore # Ignore torch import errors
from model.find_model import find_model
from inference import RainRemovalInference

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import tempfile
import argparse
from pathlib import Path

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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "proposed"
CKPT_DIR = "checkpoint"
CKPT_EPOCH = 600 ##800 or 1900
MODEL_READY = False
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model, MODEL_READY
    
    try:
        # Create necessary directories
        os.makedirs('temp_images', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('checkpoint', exist_ok=True)
        
        # First check if checkpoint exists
        if os.path.exists(os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth')):
            try:
                model = RainRemovalInference(
                    model_name=MODEL_NAME,
                    ckpt_dir=CKPT_DIR,
                    ckpt_epoch=CKPT_EPOCH,
                    device=DEVICE,
                    resize='original',
                )
                MODEL_READY = True
                print(f"Successfully initialized real model on {DEVICE}")
            except Exception as model_err:
                print(f"Error loading real model: {model_err}")
                # Fall back to mock model
                from mock_model import MockRainRemovalModel
                model = MockRainRemovalModel(os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth'))
                print(f"Falling back to mock model due to: {model_err}")
        else:
            print(f"Checkpoint file not found: {os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth')}")
            # Use mock model if checkpoint doesn't exist
            from mock_model import MockRainRemovalModel
            model = MockRainRemovalModel(os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth'))
            print("Using mock model since checkpoint doesn't exist")
            
    except Exception as e:
        print(f"Error during startup: {e}")
        print("Falling back to mock model")
        
        try:
            from mock_model import MockRainRemovalModel
            model = MockRainRemovalModel(os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth'))
            print(f"Successfully initialized fallback mock model")
        except Exception as fallback_err:
            print(f"Critical error initializing fallback model: {fallback_err}")
            model = None

@app.get("/status", tags=["API Status"])
async def get_status():
    """Check if the model is ready"""
    return JSONResponse({
        "model_ready": MODEL_READY,
        "model_type": "real" if MODEL_READY else "mock",
        "device": DEVICE
    })

@app.post("/process", tags=["Rain Removal"])
async def process_image(file: UploadFile = File(...), upscale: bool = True):
    """Process an image to remove rain streaks.
    
    Parameters:
    - file: The input image file
    - upscale: Whether to upscale the output image (default: true)
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"Processing image of size {img.shape}")
        
        # Process the image with the loaded model
        if model is None:
            from mock_model import MockRainRemovalModel
            mock_model = MockRainRemovalModel(os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth'))
            result = mock_model.process_image(img)
        elif hasattr(model, 'process_image'):
            # Use upscale option if real model and upscale is requested
            if MODEL_READY and upscale:
                result = model.process_image(img, output_size='upscale')
            else:
                result = model.process_image(img)
        else:
            result = model.process_image(img)
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Convert to correct format (ensure uint8)
        if not isinstance(result, np.ndarray):
            raise HTTPException(status_code=500, detail="Model returned invalid result type")
            
        if result.dtype != np.uint8:
            result = result.astype(np.uint8)
        
        # Determine best image quality for output
        img_quality = 90  # Higher quality JPEG
        
        # Convert to JPEG for response
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, img_quality])  # type: ignore
        result_bytes = buffer.tobytes()
        
        # Return the processed image with proper Content-Type
        return Response(
            content=result_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=processed_image.jpg",
                "X-Processing-Time": f"{processing_time:.2f}",
                "X-Model-Type": "real" if MODEL_READY else "mock",
                "X-Image-Size": f"{result.shape[1]}x{result.shape[0]}",
                "Access-Control-Expose-Headers": "Content-Disposition, X-Processing-Time, X-Model-Type, X-Image-Size"
            }
        )
                       
                       
    except HTTPException as he:
        raise he
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error processing image: {str(e)}")
        
        # Try to serve a mock result as fallback if standard processing fails
        try:
            if img is not None:
                from mock_model import MockRainRemovalModel
                fallback_model = MockRainRemovalModel(os.path.join(CKPT_DIR, f'model_epoch{CKPT_EPOCH}.pth'))
                fallback_result = fallback_model.process_image(img)
                
                # Convert to JPEG
                _, buffer = cv2.imencode('.jpg', fallback_result)
                result_bytes = buffer.tobytes()
                
                print("Served fallback result due to processing error")
                
                return Response(
                    content=result_bytes,
                    media_type="image/jpeg",
                    headers={
                        "Content-Disposition": "inline; filename=fallback_processed.jpg",
                        "X-Fallback-Result": "true",
                        "X-Model-Type": "mock",
                        "Access-Control-Expose-Headers": "Content-Disposition, X-Fallback-Result, X-Model-Type"
                    }
                )
                
        except Exception as fallback_err:
            print(f"Error creating fallback result: {fallback_err}")
            
        # If everything fails, raise the original exception
        raise HTTPException(status_code=500, detail=str(e)) 
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
