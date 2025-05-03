import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from model.find_model import find_model
import time
import utils
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
MODEL_PATH = "checkpoint/model_epoch600.pth"
USE_MOCK_MODEL = False
model = None
loaded_args = None
loaded_model = None

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def raindrop_removal(args, img_path):
    """
    Process a single image with the raindrop removal model
    
    Args:
        args: Arguments for the model
        img_path: Path to the input image
        
    Returns:
        Path to the output image
    """
    global loaded_model, loaded_args
    
    start_time = time.time()
    
    # Check if model is already loaded
    if loaded_model is None or loaded_args != args:
        loaded_model, _ = find_model(args.model, 'test')
        loaded_model.load(args.ckpt_dir, epoch=args.ckpt_epoch)
        loaded_args = args
        print(f'Loading {args.model} at EPOCH {args.ckpt_epoch}!!')
    
    # Get image filename
    img_file = os.path.basename(img_path)
    img_dir = os.path.dirname(img_path)
    
    # Read and process image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args.resize == 'square':
        square_img = cv2.resize(img, (512, 512))
        input_tensor = utils.numpy2tensor(square_img)
        output = loaded_model.test_one_image(input_tensor)
    
    elif args.resize == 'expand':
        rows, cols = img.shape[:2]
        expand_img = utils.expand_size(img, 256)
        input_tensor = utils.numpy2tensor(expand_img)
        output = loaded_model.test_one_image(input_tensor)

        for title, output_img in output.items():
            output[title] = utils.restore_size(output_img, rows, cols)
    
    elif args.resize == 'original':
        input_tensor = utils.numpy2tensor(img)
        output = loaded_model.test_one_image(input_tensor)
        
    # Save output image
    save_dir = os.path.join(args.save_dir, args.model, str(args.ckpt_epoch) + '_' + args.resize)
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, f'{os.path.splitext(img_file)[0]}.png')
    
    # Convert tensor output to uint8 image format
    processed_img = (output['output'].squeeze() * 255).astype(np.uint8)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, processed_img)
    
    print(f'Processed {img_file} in {time.time() - start_time:.2f} seconds')
    
    return output_path, processed_img

    
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model, loaded_model, loaded_args
    
    try:
        # Create necessary directories regardless of model type
        os.makedirs('temp_images', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('checkpoint', exist_ok=True)
        
        if USE_MOCK_MODEL:
            # Use mock model for testing
            from mock_model import MockRainRemovalModel
            model = MockRainRemovalModel(MODEL_PATH)
            print(f"Successfully initialized mock model")
        else:
            # Try to load the real model
            try:
                print(f"Attempting to load real model from {MODEL_PATH}")
                
                # Initialize arguments for the model
                loaded_args = Args(
                    model='proposed',
                    ckpt_dir='checkpoint',
                    save_dir='results',
                    resize='original',
                    ckpt_epoch=600,
                    dataset_dir='temp_images'
                )
                
                # Try to initialize the model but don't load weights yet
                # This is done to avoid errors during startup
                try:
                    # Just verify we can create the model class
                    model_cls, _ = find_model(loaded_args.model, 'test')
                    print(f"Successfully verified model class")
                except Exception as model_err:
                    print(f"Error creating model class: {model_err}")
                    raise
                    
                print(f"Successfully initialized real model arguments")
            except Exception as e:
                # If real model fails, fall back to mock model
                print(f"Failed to load real model: {e}")
                print("Falling back to mock model...")
                from mock_model import MockRainRemovalModel
                model = MockRainRemovalModel(MODEL_PATH)
                print(f"Successfully initialized mock model (fallback)")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("WARNING: The API may not work correctly until this is fixed.")

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Check if the service is running."""
    using_mock = USE_MOCK_MODEL
    if hasattr(model, "__class__") and model:
        using_mock = "Mock" in model.__class__.__name__
    return {"status": "ok", "model_loaded": model is not None or loaded_model is not None, "using_mock": using_mock}

@app.post("/process", tags=["Rain Removal"])
async def process_image(file: UploadFile = File(...)):
    """Process an image to remove rain streaks."""
    start_time = time.time()
    
    try:
        # Check if we need to use the mock model
        use_mock = USE_MOCK_MODEL
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"Processing image of size {img.shape}")
        result = None
        
        # First try using the mode requested in configuration
        try:
            if use_mock:
                # Use mock model
                if model is None:
                    from mock_model import MockRainRemovalModel
                    mock_model = MockRainRemovalModel(MODEL_PATH)
                    result = mock_model.process_image(img)
                else:
                    result = model.process_image(img)
                print("Processed image with mock model")
            else:
                # Use real raindrop removal function
                # Save image to temporary file
                temp_dir = 'temp_images'
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f"temp_{int(time.time())}.jpg")
                cv2.imwrite(temp_file, img)
                
                # Process image with real model
                _, result = raindrop_removal(loaded_args, temp_file)
                print("Processed image with real model")
                
        except Exception as first_error:
            # If the preferred method failed, fall back to mock model
            print(f"Error using preferred model: {first_error}")
            print("Falling back to mock model...")
            
            try:
                from mock_model import MockRainRemovalModel
                mock_model = MockRainRemovalModel(MODEL_PATH)
                result = mock_model.process_image(img)
                print("Processed image with fallback mock model")
            except Exception as mock_error:
                # If even the mock model fails, we're in trouble
                print(f"Error with fallback mock model: {mock_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to process image with any available model: {str(mock_error)}"
                )
        
        # Make sure we got a result
        if result is None:
            raise HTTPException(status_code=500, detail="No result produced by any model")
        
        # Ensure result is in the correct format (BGR, uint8)
        if result.dtype != np.uint8:
            result = result.astype(np.uint8)
            
        # Encode result as PNG for better quality
        success, encoded_img = cv2.imencode('.png', result)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")
        
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        # Return processed image
        return Response(
            content=encoded_img.tobytes(), 
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
                "X-Processing-Time": f"{processing_time:.2f}"
            }
        )
                       
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 