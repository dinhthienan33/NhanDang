# Rain Removal Model ONNX Conversion

This guide explains how to convert the PyTorch rain removal model to ONNX format and use it for faster inference.

## Why ONNX?

ONNX (Open Neural Network Exchange) provides several advantages over using PyTorch models directly:

1. **Faster inference** - ONNX Runtime is optimized for inference performance
2. **Cross-platform compatibility** - Run the same model on different hardware and platforms
3. **Hardware acceleration** - Better utilization of CPU/GPU resources
4. **Reduced dependencies** - No need for full PyTorch in production

## Requirements

Install the required packages:

```bash
pip install -r onnx_requirements.txt
```

## Converting the Model

There are two ways to convert the model:

### Option 1: Using the conversion script directly

```bash
python convert_to_onnx.py --model proposed --ckpt_dir checkpoint --ckpt_epoch 800 --output_dir onnx_models
```

Parameters:
- `--model`: Model name (default: "proposed")
- `--ckpt_dir`: Directory with checkpoint files (default: "checkpoint")
- `--ckpt_epoch`: Checkpoint epoch to load (default: 800)
- `--output_dir`: Directory to save ONNX models (default: "onnx_models")
- `--height`: Input height for ONNX model (default: 512)
- `--width`: Input width for ONNX model (default: 512)
- `--opset_version`: ONNX opset version (default: 11)

### Option 2: Using the API endpoint

With the enhanced API (`app_with_onnx.py`), you can convert the model via an HTTP request:

```bash
curl -X POST "http://localhost:8000/convert-to-onnx?model_name=proposed&ckpt_epoch=800"
```

## Using ONNX Models

### Running the API with ONNX Support

```bash
cd backend
uvicorn app_with_onnx:app --host 0.0.0.0 --port 8000
```

The API will automatically detect and use ONNX models if available.

### Using the ONNX Models Directly

You can also use the ONNX models directly with the provided wrapper:

```python
from onnx_models.onnx_inference import ONNXRainRemovalInference

# Initialize the model
model = ONNXRainRemovalInference(
    model_name="proposed",
    ckpt_dir="onnx_models",
    device="CUDA"  # or "CPU"
)

# Process an image
import cv2
img = cv2.imread("rainy_image.jpg")
result = model.process_image(img, output_size="upscale")  # or None for original size
cv2.imwrite("clean_image.jpg", result)
```

## Performance Comparison

You should see significant performance improvements when using ONNX models compared to PyTorch, especially on CPU:

- **PyTorch**: Typically 1.5-3x slower
- **ONNX**: Optimized for inference speed
- **ONNX on GPU**: May see up to 2x speed improvement over PyTorch

The API will display processing time in the response headers, which you can use to compare performance.

## Troubleshooting

1. **Error converting to ONNX**: Check that your PyTorch model works correctly first
2. **ONNX models not loading**: Ensure onnxruntime is properly installed
3. **GPU not being used**: Make sure CUDA is available and properly set up

For GPU support, ensure you have installed the CUDA-enabled version of onnxruntime:

```bash
pip install onnxruntime-gpu
``` 