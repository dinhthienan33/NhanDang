import os
import cv2  # type: ignore # Ignore cv2 import errors
import numpy as np
import torch  # type: ignore # Ignore torch import errors
from model.find_model import find_model
from inference import RainRemovalInference

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import tempfile
import argparse
from pathlib import Path

app = FastAPI(title="Rain Removal Backend with ONNX Support")

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
CKPT_EPOCH = 800  # 800 or 1900
MODEL_READY = False
model = None
ONNX_DIR = "onnx_models"
USE_ONNX = False  # Will be set to True if ONNX model is available

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model, MODEL_READY, USE_ONNX
    
    try:
        # Create necessary directories
        os.makedirs('temp_images', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('checkpoint', exist_ok=True)
        os.makedirs(ONNX_DIR, exist_ok=True)
        
        # First check if ONNX model exists
        onnx_model_exists = False
        try:
            if os.path.exists(ONNX_DIR):
                mask_models = [f for f in os.listdir(ONNX_DIR) if f.endswith('.onnx') and 'mask' in f]
                generator_models = [f for f in os.listdir(ONNX_DIR) if f.endswith('.onnx') and 'generator' in f]
                
                if len(mask_models) > 0 and len(generator_models) > 0:
                    onnx_model_exists = True
                    print(f"Found ONNX models: {mask_models[0]} and {generator_models[0]}")
        except Exception as e:
            print(f"Error checking for ONNX models: {e}")
        
        if onnx_model_exists:
            try:
                # Import ONNX inference wrapper
                import sys
                sys.path.append(ONNX_DIR)
                from onnx_inference import ONNXRainRemovalInference
                
                model = ONNXRainRemovalInference(
                    model_name=MODEL_NAME,
                    ckpt_dir=ONNX_DIR,
                    ckpt_epoch=CKPT_EPOCH,
                    device='CUDA' if DEVICE == 'cuda' else 'CPU',
                    resize='original',
                )
                MODEL_READY = True
                USE_ONNX = True
                print(f"Successfully initialized ONNX model on {DEVICE}")
            except Exception as onnx_err:
                print(f"Error loading ONNX model: {onnx_err}")
                # Fall back to PyTorch model
                onnx_model_exists = False
                print(f"Falling back to PyTorch model due to: {onnx_err}")
        
        # If ONNX model doesn't exist or failed to load, try PyTorch model
        if not onnx_model_exists:
            # Check if checkpoint exists
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
                    print(f"Successfully initialized PyTorch model on {DEVICE}")
                except Exception as model_err:
                    print(f"Error loading PyTorch model: {model_err}")
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
        "model_type": "onnx" if USE_ONNX else ("real" if MODEL_READY else "mock"),
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
                "X-Model-Type": "onnx" if USE_ONNX else ("real" if MODEL_READY else "mock"),
                "X-Image-Size": f"{result.shape[1]}x{result.shape[0]}",
                "Access-Control-Expose-Headers": "Content-Disposition, X-Processing-Time, X-Model-Type, X-Image-Size"
            }
        )
                       
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

@app.post("/convert-to-onnx", tags=["Model Management"])
async def convert_model_to_onnx(
    model_name: str = Query(default="proposed", description="Model name"),
    ckpt_epoch: int = Query(default=800, description="Checkpoint epoch"),
    height: int = Query(default=512, description="Input height for ONNX model"),
    width: int = Query(default=512, description="Input width for ONNX model"),
    opset_version: int = Query(default=11, description="ONNX opset version")
):
    """Convert PyTorch model to ONNX format for better inference."""
    try:
        from convert_to_onnx import convert_to_onnx
        
        output_dir = ONNX_DIR
        
        # Check if PyTorch model exists
        if not os.path.exists(os.path.join(CKPT_DIR, f'model_epoch{ckpt_epoch}.pth')):
            return JSONResponse(
                status_code=404,
                content={"detail": f"Checkpoint file not found: model_epoch{ckpt_epoch}.pth"}
            )
        
        # Convert model to ONNX
        generator_path, mask_path = convert_to_onnx(
            model_name=model_name,
            ckpt_dir=CKPT_DIR,
            ckpt_epoch=ckpt_epoch,
            output_dir=output_dir,
            input_shape=(1, 3, height, width),
            opset_version=opset_version
        )
        
        # Return success response
        return JSONResponse({
            "status": "success",
            "message": "Model converted to ONNX format successfully",
            "models": {
                "generator": generator_path,
                "mask": mask_path
            },
            "note": "Restart the API to use the ONNX model for inference"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error converting model to ONNX: {str(e)}"}
        )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_with_onnx:app", host="0.0.0.0", port=8000, reload=True) 